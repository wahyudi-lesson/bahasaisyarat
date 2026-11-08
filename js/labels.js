/**
 * js/labels.js
 * Manajemen pemetaan label output model.
 * Memuat secara dinamis dari model_web/labels.json dengan fallback cadangan.
 */

let SIGN_LABELS = {};

// Fallback jika labels.json belum sempat termuat
const FALLBACK_LABELS = {
  0: "A", 1: "B", 2: "C"
};

/**
 * Memuat mapping label secara otomatis dari berkas model_web/labels.json
 */
async function loadSignLabels() {
  try {
    const res = await fetch('model_web/labels.json?t=' + Date.now()); // anti-cache
    if (!res.ok) throw new Error("Gagal mengambil labels.json");
    SIGN_LABELS = await res.json();
    console.log("Labels berhasil dimuat dari labels.json:", SIGN_LABELS);
  } catch (error) {
    console.warn("Menggunakan fallback label sementara:", error);
    SIGN_LABELS = FALLBACK_LABELS;
  }
}

/**
 * Mengambil nama label berdasarkan indeks prediksi
 * @param {number} index
 * @returns {string}
 */
function getLabelByIndex(index) {
  if (SIGN_LABELS[index] !== undefined) {
    return SIGN_LABELS[index];
  }
  return `Kelas #${index}`;
}

// Ekspos ke global window
if (typeof window !== 'undefined') {
  window.SIGN_LABELS = SIGN_LABELS;
  window.getLabelByIndex = getLabelByIndex;
  window.loadSignLabels = loadSignLabels;
}
