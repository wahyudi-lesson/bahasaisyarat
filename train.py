import os
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("=== MEMULAI PELATIHAN OTOMATIS GITHUB ACTIONS ===")

# 1. Muat dataset
csv_path = 'dataset_landmarks.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError("dataset_landmarks.csv tidak ditemukan!")

df = pd.read_csv(csv_path)
X_raw = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

# 2. Encode label dan simpan labels.json
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)

labels_dict = {int(i): str(label) for i, label in enumerate(encoder.classes_)}
os.makedirs('model_web', exist_ok=True)
with open('model_web/labels.json', 'w') as f:
    json.dump(labels_dict, f, indent=2)

print(f"Dataset terbaca: {len(X_raw)} baris, {num_classes} kelas: {list(encoder.classes_)}")

# 3. Augmentasi Sintetis (Jittering & Scaling)
def augment_landmarks(landmarks_vector, num_copies=40):
    augmented = []
    coords = landmarks_vector.reshape(42, 3) # 42 titik (2 tangan x 21)
    for _ in range(num_copies):
        scale = np.random.uniform(0.96, 1.04)
        noise = np.random.normal(0, 0.01, coords.shape)
        augmented.append(((coords * scale) + noise).flatten())
    return augmented

X_list, y_list = [], []
for i in range(len(X_raw)):
    X_list.append(X_raw[i])
    y_list.append(y_encoded[i])
    for synth in augment_landmarks(X_raw[i], num_copies=40):
        X_list.append(synth)
        y_list.append(y_encoded[i])

X = np.array(X_list, dtype=np.float32)
y = tf.keras.utils.to_categorical(np.array(y_list), num_classes=num_classes)

# 4. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Bangun Model (Input 126 fitur)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(126,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Training
model.fit(X_train, y_train, epochs=35, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 7. Ekspor Bobot ke Format Binary & JSON (Bebas tensorflowjs pip bug)
weights_manifest = []
bin_bytes = bytearray()
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
    "modelTopology": {
        "keras_version": "2.15.0",
        "backend": "tensorflow",
        "model_config": {
            "class_name": "Sequential",
            "config": {
                "name": "sign_language_model",
                "layers": [
                    {"class_name": "InputLayer", "config": {"batch_input_shape": [None, 126], "dtype": "float32", "sparse": False, "name": "input_coords"}},
                    {"class_name": "Dense", "config": {"name": "dense_1", "units": 128, "activation": "relu", "use_bias": True}},
                    {"class_name": "Dropout", "config": {"name": "dropout_1", "rate": 0.2}},
                    {"class_name": "Dense", "config": {"name": "dense_2", "units": 64, "activation": "relu", "use_bias": True}},
                    {"class_name": "Dropout", "config": {"name": "dropout_2", "rate": 0.2}},
                    {"class_name": "Dense", "config": {"name": "dense_3", "units": num_classes, "activation": "softmax", "use_bias": True}}
                ]
            }
        }
    },
    "weightsManifest": [{"paths": [bin_filename], "weights": weights_manifest}]
}

with open(os.path.join('model_web', 'model.json'), 'w') as f:
    json.dump(model_json_content, f, indent=2)

print("=== MODEL BERHASIL DIEKSPOR KE model_web/ ===")
