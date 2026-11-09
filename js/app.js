const videoElement = document.getElementById('webcam') || document.querySelector('video');
const canvasElement = document.getElementById('output_canvas') || document.querySelector('canvas');
const canvasCtx = canvasElement.getContext('2d');

const currentPredictionEl = document.getElementById('current-sign');
const accuracyEl = document.getElementById('accuracy-val');
const sentenceEl = document.getElementById('sentence-text');
const btnSpeak = document.getElementById('btn-speak');
const btnClear = document.getElementById('btn-clear');
const btnToggleMode = document.getElementById('btn-toggle-mode');
const modelStatusEl = document.getElementById('ai-model-status');

let model = null;
let isModelReady = false;
let isAutoSpeak = localStorage.getItem('isAutoSpeak') === 'true';

let candidateLabel = "";
let candidateCount = 0;
let lastConfirmedLabel = "";
let noHandFrames = 0;

const CONSECUTIVE_THRESHOLD = 7;
const CONFIDENCE_THRESHOLD = 0.70;
const NO_HAND_RESET_FRAMES = 12;

let sentenceBuffer = [];

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

// Inisialisasi Model AI dengan cache buster
async function initModel() {
  try {
    if (modelStatusEl) {
      modelStatusEl.textContent = "⏳ Memuat model AI dari repositori...";
      modelStatusEl.style.color = "#facc15";
    }

    await window.loadSignLabels();
    model = await tf.loadLayersModel(`model_web/model.json?t=${Date.now()}`);
    isModelReady = true;

    const classNames = Object.values(window.SIGN_LABELS || {}).join(', ');
    if (modelStatusEl) {
      modelStatusEl.textContent = `✅ AI Aktif (${Object.keys(window.SIGN_LABELS || {}).length} Kelas: ${classNames})`;
      modelStatusEl.style.color = "#34d399";
    }
  } catch (err) {
    console.error("Gagal memuat model:", err);
    isModelReady = false;
    if (modelStatusEl) {
      modelStatusEl.textContent = "❌ Model AI belum siap: " + err.message;
      modelStatusEl.style.color = "#ef4444";
    }
  }
}
initModel();

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
    noHandFrames = 0;
    drawSkeleton(results.multiHandLandmarks);

    if (isModelReady && window.normalizeLandmarks126) {
      const inputVector = window.normalizeLandmarks126(results);
      if (inputVector) {
        predictGesture(inputVector);
      }
    }
  } else {
    noHandFrames++;
    if (noHandFrames >= NO_HAND_RESET_FRAMES) {
      lastConfirmedLabel = "";
      candidateLabel = "";
      candidateCount = 0;
    }
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
  }
});

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

    if (maxProb >= CONFIDENCE_THRESHOLD) {
      if (detectedLabel === candidateLabel) {
        candidateCount++;
        if (candidateCount >= CONSECUTIVE_THRESHOLD && detectedLabel !== lastConfirmedLabel) {
          lastConfirmedLabel = detectedLabel;
          appendWord(detectedLabel);
          if (isAutoSpeak) speakText(detectedLabel);
        }
      } else {
        candidateLabel = detectedLabel;
        candidateCount = 1;
      }
    } else {
      candidateCount = 0;
    }
  });
}

function appendWord(word) {
  sentenceBuffer.push(word);
  if (sentenceEl) sentenceEl.textContent = sentenceBuffer.join(' ');
}

function speakText(text) {
  if (!text || !('speechSynthesis' in window)) return;
  const readable = text.replace(/_/g, ' ').toLowerCase();
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(readable);
  utterance.lang = 'id-ID';
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

if (btnSpeak) {
  btnSpeak.onclick = () => {
    const full = sentenceBuffer.join(' ');
    speakText(full || (currentPredictionEl ? currentPredictionEl.textContent : ""));
  };
}

if (btnClear) {
  btnClear.onclick = () => {
    sentenceBuffer = [];
    if (sentenceEl) sentenceEl.textContent = "";
    if (currentPredictionEl) currentPredictionEl.textContent = "-";
    if (accuracyEl) accuracyEl.textContent = "Akurasi: 0%";
    candidateCount = 0;
    candidateLabel = "";
    lastConfirmedLabel = "";
    window.speechSynthesis.cancel();
  };
}

const camera = new Camera(videoElement, {
  onFrame: async () => { await hands.send({ image: videoElement }); },
  width: 640,
  height: 480,
  facingMode: "user"
});
camera.start();
