/**
 * js/app.js
 * Logika utama deteksi webcam, inferensi TensorFlow.js (126 fitur),
 * state lock pencegah repetisi, dan sintesis suara (TTS) stabil.
 */

const videoElement = document.getElementById('webcam') || document.querySelector('video');
const canvasElement = document.getElementById('output_canvas') || document.querySelector('canvas');
const canvasCtx = canvasElement.getContext('2d');

// Elemen UI
const currentPredictionEl = document.getElementById('current-sign');
const accuracyEl = document.getElementById('accuracy-val');
const sentenceEl = document.getElementById('sentence-text');
const btnSpeak = document.getElementById('btn-speak');
const btnClear = document.getElementById('btn-clear');
const btnToggleMode = document.getElementById('btn-toggle-mode');

let model = null;
let isModelReady = false;

// Mode Suara: false = Manual, true = Otomatis Langsung Baca
let isAutoSpeak = localStorage.getItem('isAutoSpeak') === 'true';

// Variabel Kontrol dan Debounce Cerdas
let candidateLabel = "";
let candidateCount = 0;
let lastConfirmedLabel = ""; // Pengunci agar tidak mengulang kata yang sama
let noHandFrames = 0;

const CONSECUTIVE_THRESHOLD = 7;     // Butuh 7 frame konsisten (~0.25 detik)
const CONFIDENCE_THRESHOLD = 0.75;    // Minimal keyakinan model 75%
const NO_HAND_RESET_FRAMES = 12;     // Butuh jeda ~0.4 detik tanpa tangan untuk mereset huruf yang sama

let sentenceBuffer = [];

// Perbarui visual tombol mode suara
function updateModeUI() {
  if (!btnToggleMode) return;
  if (isAutoSpeak) {
    btnToggleMode.className = "btn-mode-toggle auto-mode";
    btnToggleMode.innerHTML = "Mode Suara: ⚡ Langsung Baca (Otomatis)";
    if (btnSpeak) btnSpeak.textContent = "🔊 Ulangi Kalimat";
  } else {
    btnToggleMode.className = "btn-mode-toggle manual-mode";
    btnToggleMode.innerHTML = "Mode Suara: 📝 Kumpulkan Kata (Klik Ucapkan)";
    if (btnSpeak) btnSpeak.textContent = "🔊 Ucapkan Suara";
  }
}
updateModeUI();

if (btnToggleMode) {
  btnToggleMode.onclick = () => {
    isAutoSpeak = !isAutoSpeak;
    localStorage.setItem('isAutoSpeak', isAutoSpeak);
    updateModeUI();
  };
}

// 1. Inisialisasi Model AI & Labels
async function initModel() {
  try {
    console.log("Memuat model dan daftar label...");
    await window.loadSignLabels();
    model = await tf.loadLayersModel('model_web/model.json');
    isModelReady = true;
    console.log("Model AI berhasil dimuat!");
  } catch (err) {
    console.error("Gagal memuat model:", err);
  }
}
initModel();

// 2. Hubungkan Titik Sendi MediaPipe
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

function drawSkeleton(landmarksList) {
  landmarksList.forEach(landmarks => {
    canvasCtx.strokeStyle = '#22c55e';
    canvasCtx.lineWidth = 3;
    HAND_CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      canvasCtx.beginPath();
      canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
      canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
      canvasCtx.stroke();
    });

    landmarks.forEach(pt => {
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvasElement.width, pt.y * canvasElement.height, 4, 0, 2 * Math.PI);
      canvasCtx.fillStyle = '#ef4444';
      canvasCtx.fill();
    });
  });
}

// 3. Inisialisasi MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(async (results) => {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    noHandFrames = 0; // Ada tangan di kamera
    drawSkeleton(results.multiHandLandmarks);

    if (isModelReady && window.normalizeLandmarks126) {
      const inputVector = window.normalizeLandmarks126(results);
      if (inputVector) {
        predictGesture(inputVector);
      }
    }
  } else {
    // Tidak ada tangan di frame: hitung jeda untuk reset kata yang sama
    noHandFrames++;
    if (noHandFrames >= NO_HAND_RESET_FRAMES) {
      lastConfirmedLabel = ""; // Bolehkan mengulang huruf yang sama jika tangan sempat turun
      candidateLabel = "";
      candidateCount = 0;
    }
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
  }
});

// 4. Inferensi Model AI dengan Pencegah Looping
async function predictGesture(features126) {
  tf.tidy(() => {
    const inputTensor = tf.tensor2d([features126], [1, 126]);
    const output = model.predict(inputTensor);
    const probabilities = output.dataSync();

    let maxProb = 0;
    let maxIndex = 0;
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    const detectedLabel = window.getLabelByIndex(maxIndex);
    const confidencePct = (maxProb * 100).toFixed(1);

    if (currentPredictionEl) currentPredictionEl.textContent = detectedLabel;
    if (accuracyEl) accuracyEl.textContent = `Akurasi: ${confidencePct}%`;

    // Filter Kestabilan Pose
    if (maxProb >= CONFIDENCE_THRESHOLD) {
      if (detectedLabel === candidateLabel) {
        candidateCount++;
        
        // Cek apakah pose sudah cukup lama stabil DAN belum pernah dikonfirmasi sebelumnya
        if (candidateCount >= CONSECUTIVE_THRESHOLD && detectedLabel !== lastConfirmedLabel) {
          lastConfirmedLabel = detectedLabel; // KUNCI! Tidak akan dieksekusi lagi selama tangan diam
          appendWord(detectedLabel);

          if (isAutoSpeak) {
            speakText(detectedLabel);
          }
        }
      } else {
        // Posisi tangan mulai berubah membentuk pose lain
        candidateLabel = detectedLabel;
        candidateCount = 1;
      }
    } else {
      candidateCount = 0;
    }
  });
}

// 5. Pengelolaan Kalimat & Suara (Text-to-Speech)
function appendWord(word) {
  sentenceBuffer.push(word);
  renderSentence();
}

function renderSentence() {
  if (sentenceEl) {
    sentenceEl.textContent = sentenceBuffer.join(' ');
  }
}

function speakText(text) {
  if (!text || !('speechSynthesis' in window)) return;

  const readableText = text.replace(/_/g, ' ').toLowerCase();

  // Hentikan audio sebelumnya hanya jika kata yang dibaca benar-benar kata baru
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(readableText);
  utterance.lang = 'id-ID';
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// Tombol Ucapkan Manual
if (btnSpeak) {
  btnSpeak.onclick = () => {
    const fullText = sentenceBuffer.join(' ');
    speakText(fullText || (currentPredictionEl ? currentPredictionEl.textContent : ""));
  };
}

// Tombol Hapus Kalimat
if (btnClear) {
  btnClear.onclick = () => {
    sentenceBuffer = [];
    renderSentence();
    if (currentPredictionEl) currentPredictionEl.textContent = "-";
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
    candidateCount = 0;
    candidateLabel = "";
    lastConfirmedLabel = "";
    window.speechSynthesis.cancel();
  };
}

// 6. Inisialisasi Kamera Depan
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480
});
camera.start();
