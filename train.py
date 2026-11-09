import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("=== MEMULAI PELATIHAN DATASET ISYARAT ===")

csv_path = 'dataset_landmarks.csv'
if not os.path.exists(csv_path):
    print("dataset_landmarks.csv belum ada. Pelatihan dihentikan.")
    exit(0)

# 1. Baca CSV dan tangani karakter baris baru
df = pd.read_csv(csv_path)
df.dropna(inplace=True)

X_raw = df.iloc[:, :126].values.astype(np.float32)
y_raw = df.iloc[:, -1].astype(str).str.strip().values

# 2. Encode label kelas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)

os.makedirs('model_web', exist_ok=True)
labels_dict = {str(i): str(label) for i, label in enumerate(encoder.classes_)}
with open('model_web/labels.json', 'w') as f:
    json.dump(labels_dict, f, indent=2)

print(f"Total Sampel: {len(X_raw)} baris | Jumlah Kelas: {num_classes} ({list(encoder.classes_)})")

# 3. Augmentasi Sintetis Koordinat
def augment_landmarks(landmarks_vector, num_copies=35):
    augmented = []
    coords = landmarks_vector.reshape(42, 3)
    for _ in range(num_copies):
        scale = np.random.uniform(0.96, 1.04)
        noise = np.random.normal(0, 0.01, coords.shape)
        augmented.append(((coords * scale) + noise).flatten())
    return augmented

X_list, y_list = [], []
for i in range(len(X_raw)):
    X_list.append(X_raw[i])
    y_list.append(y_encoded[i])
    for synth in augment_landmarks(X_raw[i], num_copies=35):
        X_list.append(synth)
        y_list.append(y_encoded[i])

X = np.array(X_list, dtype=np.float32)
y = tf.keras.utils.to_categorical(np.array(y_list), num_classes=num_classes)

# 4. Arsitektur Model (Input shape langsung di Dense pertama)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(126,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Latih Model
if len(X) > 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=1)
else:
    model.fit(X, y, epochs=10, batch_size=4, verbose=1)

# 5. Ekspor Bobot ke Binary & JSON yang Kompatibel Penuh dengan TF.js Browser
bin_bytes = bytearray()
weights_manifest = []
dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

for idx, layer in enumerate(dense_layers):
    layer_name = f"dense_{idx + 1}"
    w, b = layer.get_weights()
    weights_manifest.append({"name": f"{layer_name}/kernel", "shape": list(w.shape), "dtype": "float32"})
    bin_bytes.extend(w.astype('<f4').tobytes())
    weights_manifest.append({"name": f"{layer_name}/bias", "shape": list(b.shape), "dtype": "float32"})
    bin_bytes.extend(b.astype('<f4').tobytes())

bin_filename = 'group1-shard1of1.bin'
with open(os.path.join('model_web', bin_filename), 'wb') as f:
    f.write(bin_bytes)

model_json_content = {
    "format": "layers-model",
    "generatedBy": "GitHub Actions Auto-Trainer",
    "convertedBy": None,
    "modelTopology": {
        "keras_version": "2.15.0",
        "backend": "tensorflow",
        "model_config": {
            "class_name": "Sequential",
            "config": {
                "name": "sign_language_model",
                "layers": [
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "dense_1",
                            "trainable": True,
                            "batch_input_shape": [None, 126],
                            "dtype": "float32",
                            "units": 128,
                            "activation": "relu",
                            "use_bias": True
                        }
                    },
                    {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": True, "rate": 0.2}},
                    {"class_name": "Dense", "config": {"name": "dense_2", "trainable": True, "units": 64, "activation": "relu", "use_bias": True}},
                    {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": True, "rate": 0.2}},
                    {"class_name": "Dense", "config": {"name": "dense_3", "trainable": True, "units": num_classes, "activation": "softmax", "use_bias": True}}
                ]
            }
        }
    },
    "weightsManifest": [{"paths": [bin_filename], "weights": weights_manifest}]
}

with open(os.path.join('model_web', 'model.json'), 'w') as f:
    json.dump(model_json_content, f, indent=2)

print("=== PELATIHAN SUKSES: model_web/ TELAH DIPERBARUI ===")
