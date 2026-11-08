# 🖐️ Web Sign Language Recognition & Text-to-Speech

Aplikasi web *real-time* berbasis browser untuk mendeteksi bahasa isyarat menggunakan webcam, menerjemahkannya ke teks, dan membunyikan suaranya dalam Bahasa Indonesia secara otomatis.

Sistem ini berjalan 100% di sisi klien (*client-side*) menggunakan **MediaPipe Hands**, **TensorFlow.js**, dan **Web Speech API**, sehingga dapat dihosting gratis di **GitHub Pages** tanpa memerlukan server backend Python saat produksi.

---

## ✨ Fitur Utama

- **Real-Time Landmark Detection:** Melacak 21 titik sendi tangan menggunakan MediaPipe Hands (WASM/WebGL).
- **Client-Side Inference:** Menjalankan model klasifikasi langsung di browser menggunakan TensorFlow.js tanpa latensi jaringan.
- **Scale & Translation Invariant:** Koordinat dinormalisasi terhadap titik pergelangan tangan (*wrist*) sehingga akurat pada berbagai jarak kamera.
- **Debounced Text-to-Speech:** Menggunakan algoritma *history buffer* untuk mencegah pengulangan suara (*flickering*) sebelum membacakan teks via Web Speech API (`id-ID`).
- **Zero Server Cost:** Sepenuhnya statis dan di-deploy otomatis via GitHub Actions ke GitHub Pages.

---

## 📁 Struktur Repositori

```text
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions workflow untuk auto-deploy Pages
├── index.html                   # Halaman antarmuka utama (Webcam, canvas, teks)
├── css/
│   └── style.css                # Desain UI antarmuka responsif
├── js/
│   ├── app.js                   # Kontrol loop kamera, inferensi, dan Text-to-Speech
│   ├── normalizer.js            # Algoritma normalisasi 21 koordinat landmark
│   └── labels.js                # Pemetaan indeks prediksi model ke karakter
├── model_web/                   # Model hasil konversi TensorFlow.js
│   ├── model.json               # Topologi graf neural network
│   ├── group1-shard1of1.bin     # Bobot biner tensor (weights)
│   └── labels.json              # Cadangan daftar mapping kelas
├── training/                    # Lingkungan kerja Python (Data & Training)
│   ├── collect_data.py          # Skrip lokal pengambil koordinat webcam ke CSV
│   ├── dataset_landmarks.csv    # Dataset tabular 63 fitur koordinat + label
│   └── train_colab.ipynb        # Notebook pelatihan MLP di Google Colab
├── .gitignore
└── README.md
