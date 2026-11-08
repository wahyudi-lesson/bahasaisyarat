/**
 * js/app.js
 * Logika utama deteksi webcam, inferensi TensorFlow.js (126 fitur),
 * visualisasi kerangka tangan ganda, dan sintesis suara (TTS).
 */

const videoElement = document.getElementById('webcam') || document.querySelector('video');
const canvasElement = document.getElementById('output_canvas') || document.querySelector('canvas');
const canvasCtx = canvasElement.getContext('2d');

// Elemen UI
const currentPredictionEl = document.getElementById('current-sign') || document.querySelector('.prediction-char') || document.querySelector('h2');
const accuracyEl = document.getElementById('accuracy-val') || document.querySelector('.accuracy-value');
const sentenceEl = document.getElementById('sentence-text') || document.querySelector('.sentence-box');
const btnSpeak = document.getElementById('btn-speak') || document.querySelector('button[onclick*="speak"]');
const btnClear = document.getElementById('btn-clear') || document.querySelector('button[onclick*="clear"]');

let model = null;
let isModelReady = false;

// Variabel stabilisasi deteksi (Debounce)
let consecutiveCount = 0;
let lastDetectedChar = "";
const CONSECUTIVE_THRESHOLD = 8; // Harus konsisten selama 8 frame
const CONFIDENCE_THRESHOLD = 0.75; // Minimal akurasi 75%
let sentenceBuffer = [];

// 1. Inisialisasi Model TensorFlow.js & Labels
async function initModel() {
  try {
    console.log("Memuat model dan daftar label...");
    await window.loadSignLabels();
    model = await tf.loadLayersModel('model_web/model.json');
    isModelReady = true;
    console.log("Model AI berhasil dimuat!");
  } catch (err) {
    console.error("Gagal memuat model:", err);
    alert("Gagal memuat model_web/model.json. Pastikan berkas sudah diunggah ke GitHub.");
  }
}
initModel();

// 2. Hubungkan Titik Sendi MediaPipe (Koneksi Kerangka Tangan)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Ibu jari
  [0, 5], [5, 6], [6, 7], [7, 8],       // Telunjuk
  [5, 9], [9, 10], [10, 11], [11, 12],  // Jari tengah
  [9, 13], [13, 14], [14, 15], [15, 16],// Jari manis
  [13, 17], [17, 18], [18, 19], [19, 20],// Kelingking
  [0, 17]                               // Telapak bawah
];

function drawSkeleton(landmarksList) {
  landmarksList.forEach(landmarks => {
    // Gambar garis penghubung hijau
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

    // Gambar titik sendi merah
    landmarks.forEach(pt => {
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvasElement.width, pt.y * canvasElement.height, 4, 0, 2 * Math.PI);
      canvasCtx.fillStyle = '#ef4444';
      canvasCtx.fill();
    });
  });
}

// 3. Inisialisasi MediaPipe Hands (Maksimal 2 Tangan)
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
  // Samakan ukuran kanvas dengan video feed
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    // Gambar kerangka tangan di kanvas
    drawSkeleton(results.multiHandLandmarks);

    // Lakukan prediksi hanya jika model sudah siap
    if (isModelReady && window.normalizeLandmarks126) {
      const inputVector = window.normalizeLandmarks126(results);

      if (inputVector) {
        predictGesture(inputVector);
      }
    }
  } else {
    // Reset status jika tangan keluar dari frame kamera
    consecutiveCount = 0;
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
  }
});

// 4. Proses Inferensi Model AI
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

    // Perbarui UI karakter & akurasi
    if (currentPredictionEl) currentPredictionEl.textContent = detectedLabel;
    if (accuracyEl) accuracyEl.textContent = `Akurasi: ${confidencePct}%`;

    // Filter stabilisasi: Simpan kata jika posisi stabil di atas ambang batas
    if (maxProb >= CONFIDENCE_THRESHOLD) {
      if (detectedLabel === lastDetectedChar) {
        consecutiveCount++;
        if (consecutiveCount === CONSECUTIVE_THRESHOLD) {
          appendWord(detectedLabel);
          consecutiveCount = 0; // reset counter
        }
      } else {
        lastDetectedChar = detectedLabel;
        consecutiveCount = 1;
      }
    } else {
      consecutiveCount = 0;
    }
  });
}

// 5. Rangkaian Kata & Logika Text-to-Speech
function appendWord(charOrWord) {
  sentenceBuffer.push(charOrWord);
  renderSentence();
}

function renderSentence() {
  if (sentenceEl) {
    sentenceEl.textContent = sentenceBuffer.join(' ');
  }
}

function speakText(text) {
  if (!text || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel(); // Hentikan suara yang sedang berjalan

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'id-ID';
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

// Tombol Aksi
if (btnSpeak) {
  btnSpeak.onclick = () => {
    const fullText = sentenceBuffer.join(' ');
    speakText(fullText || (currentPredictionEl ? currentPredictionEl.textContent : ""));
  };
}

if (btnClear) {
  btnClear.onclick = () => {
    sentenceBuffer = [];
    renderSentence();
    if (currentPredictionEl) currentPredictionEl.textContent = "-";
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
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
