import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

CSV_FILE = 'dataset_landmarks.csv'

# Buat header CSV jika file belum ada (63 koordinat + 1 label)
if not os.path.exists(CSV_FILE):
    headers = []
    for i in range(21):
        headers.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    headers.append('label')
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    # 1. Translasi: jadikan pergelangan tangan (wrist / index 0) sebagai titik (0,0,0)
    base = coords[0]
    rel_coords = coords - base
    # 2. Penskalaan: bagi dengan jarak terjauh ke pergelangan tangan
    max_value = np.max(np.linalg.norm(rel_coords, axis=1))
    if max_value > 0:
        rel_coords = rel_coords / max_value
    return rel_coords.flatten().tolist()

cap = cv2.VideoCapture(0)
current_label = ""
is_recording = False
samples_count = 0

print("=== PENGUMPUL DATA BAHASA ISYARAT ===")
print("Tekan tombol huruf/angka di keyboard untuk menetapkan label (misal: 'A', 'B', 'C').")
print("Tekan 'SPACE' untuk mulai/berhenti merekam sampel.")
print("Tekan 'Q' untuk keluar dari program.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_recording and current_label:
                normalized_features = normalize_landmarks(hand_landmarks.landmark)
                normalized_features.append(current_label)
                
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(normalized_features)
                
                samples_count += 1

    # Status tampilan overlay
    status_text = f"Label: {current_label or '-'} | Rekam: {'AKTIF' if is_recording else 'OFF'} | Sampel: {samples_count}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_recording else (0, 0, 255), 2)
    cv2.imshow("Koleksi Data Isyarat", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Spasi
        if current_label:
            is_recording = not is_recording
    elif key != 255:
        char = chr(key).upper()
        if char.isalnum():
            current_label = char
            samples_count = 0
            is_recording = False

cap.release()
cv2.destroyAllWindows()
