let SIGN_LABELS = {};

async function loadSignLabels() {
  try {
    const res = await fetch('model_web/labels.json?t=' + Date.now());
    if (!res.ok) throw new Error("labels.json tidak ditemukan");
    const data = await res.json();
    
    // Konversi baik array maupun object ke mapping standar
    SIGN_LABELS = {};
    if (Array.isArray(data)) {
      data.forEach((val, idx) => { SIGN_LABELS[idx] = val; });
    } else {
      Object.entries(data).forEach(([k, val]) => {
        SIGN_LABELS[parseInt(k)] = val;
      });
    }
    console.log("Labels loaded:", SIGN_LABELS);
  } catch (err) {
    console.warn("Gagal memuat labels.json, menggunakan fallback:", err);
    SIGN_LABELS = { 0: "A", 1: "B", 2: "C" };
  }
}

function getLabelByIndex(index) {
  return SIGN_LABELS[index] !== undefined ? SIGN_LABELS[index] : `Kelas #${index}`;
}

if (typeof window !== 'undefined') {
  window.SIGN_LABELS = SIGN_LABELS;
  window.getLabelByIndex = getLabelByIndex;
  window.loadSignLabels = loadSignLabels;
}
