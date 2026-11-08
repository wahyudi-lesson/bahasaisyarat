/**
 * Normalisasi koordinat MediaPipe Hands ke 126 fitur terpadu:
 * - Indeks 0 s.d. 62   : Tangan Kiri (21 landmark x 3 koordinat)
 * - Indeks 63 s.d. 125 : Tangan Kanan (21 landmark x 3 koordinat)
 * Jika hanya 1 tangan yang terdeteksi, slot tangan lainnya diisi 0.0.
 */
function normalizeLandmarks126(results) {
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    return null;
  }

  let leftHandCoords = new Array(63).fill(0.0);
  let rightHandCoords = new Array(63).fill(0.0);

  results.multiHandLandmarks.forEach((landmarks, index) => {
    // MediaPipe label: 'Left' atau 'Right'
    const handedness = results.multiHandedness && results.multiHandedness[index]
      ? results.multiHandedness[index].label
      : (index === 0 ? 'Right' : 'Left');

    // 1. Normalisasi translasi: Jadikan wrist (titik 0) sebagai titik pusat (0,0,0)
    const wrist = landmarks[0];

    // 2. Normalisasi skala: Jarak dari wrist (0) ke pangkal jari tengah (MCP titik 9)
    const refJoint = landmarks[9];
    const scaleFactor = Math.hypot(
      refJoint.x - wrist.x,
      refJoint.y - wrist.y,
      refJoint.z - wrist.z
    ) || 1.0;

    const normalized = [];
    landmarks.forEach(pt => {
      normalized.push(
        (pt.x - wrist.x) / scaleFactor,
        (pt.y - wrist.y) / scaleFactor,
        (pt.z - wrist.z) / scaleFactor
      );
    });

    if (handedness === 'Left') {
      leftHandCoords = normalized;
    } else {
      rightHandCoords = normalized;
    }
  });

  return [...leftHandCoords, ...rightHandCoords];
}

// Kompatibilitas global browser
if (typeof window !== 'undefined') {
  window.normalizeLandmarks126 = normalizeLandmarks126;
}
