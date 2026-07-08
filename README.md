<div align="center">

# 🛡️ Credit Card Fraud Detection
### Deteksi Anomali Transaksi Kartu Kredit dengan Isolation Forest

<br>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

<br>

> **Proyek Machine Learning** — Mendeteksi transaksi kartu kredit yang mencurigakan menggunakan algoritma **Isolation Forest** dengan pendekatan *Unsupervised Anomaly Detection*.

<br>

---

</div>

## 👨‍💻 Identitas Penulis

<div align="center">

| 🎓 Nama | **Muhammad Sahal Anwar Hadi** |
|---|---|
| 🪪 NIM | **24260032** |
| 📚 Mata Kuliah | Machine Learning |

</div>

---

## 📋 Daftar Isi

- [🎯 Deskripsi Proyek](#-deskripsi-proyek)
- [🧠 Algoritma yang Digunakan](#-algoritma-yang-digunakan)
- [📁 Struktur Folder](#-struktur-folder)
- [📊 Dataset](#-dataset)
- [🔬 Metodologi](#-metodologi)
- [📈 Visualisasi Hasil](#-visualisasi-hasil)
- [⚙️ Cara Menjalankan](#️-cara-menjalankan)
- [📦 Dependensi](#-dependensi)
- [📉 Hasil & Evaluasi Model](#-hasil--evaluasi-model)
- [📄 Laporan & Presentasi](#-laporan--presentasi)
- [📌 Catatan Teknis](#-catatan-teknis)

---

## 🎯 Deskripsi Proyek

Proyek ini bertujuan untuk **mendeteksi transaksi kartu kredit yang bersifat fraudulens (penipuan)** menggunakan teknik **Anomaly Detection berbasis Isolation Forest**.

Penipuan kartu kredit merupakan masalah serius di industri keuangan global. Dataset yang digunakan bersifat **sangat imbalanced** (tidak seimbang), di mana transaksi fraud hanya sebagian kecil dari total transaksi. Oleh karena itu, pendekatan *unsupervised learning* seperti **Isolation Forest** dipilih karena:

- ✅ Tidak bergantung pada label data yang lengkap
- ✅ Efisien untuk data berdimensi tinggi
- ✅ Mampu mengisolasi anomali dengan cepat menggunakan struktur pohon acak
- ✅ Cocok untuk data imbalanced

---

## 🧠 Algoritma yang Digunakan

### 🌲 Isolation Forest

```
Konsep Dasar:
Anomali adalah data yang "mudah diisolasi" karena berbeda dari mayoritas.
Isolation Forest membangun banyak pohon acak dan mengukur
seberapa cepat sebuah titik data dapat diisolasi.

Semakin pendek jalur isolasi  →  semakin besar kemungkinan ANOMALI
Semakin panjang jalur isolasi →  semakin besar kemungkinan NORMAL
```

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `n_estimators` | 100 | Jumlah pohon dalam hutan |
| `contamination` | auto / custom | Proporsi perkiraan anomali |
| `max_samples` | auto | Sampel per pohon |
| `random_state` | 42 | Reproduktibilitas hasil |

**Alur Kerja Model:**

```
Input Data  →  StandardScaler  →  Isolation Forest  →  Anomaly Score
     ↓               ↓                    ↓                   ↓
  Raw CSV        Normalized         Trained Model       -1=Fraud / 1=Normal
```

---

## 📁 Struktur Folder

```
📦 Credit-Card-Fraud-Detection/
│
├── 📓 isolation_forest_anomaly.ipynb    ← Notebook utama (EDA + Training + Evaluasi)
├── 🐍 test_model.py                     ← Script pengujian model terlatih
│
├── 📊 creditcard.csv                    ← Dataset utama (284,807 transaksi)
├── 📊 test_data.csv                     ← Data uji khusus
├── 📊 test_result.csv                   ← Hasil prediksi pada data uji
│
├── 📂 output/
│   ├── 📂 model/
│   │   ├── 🤖 isolation_forest.pkl      ← Model terlatih (tersimpan)
│   │   └── ⚖️  standard_scaler.pkl      ← Scaler tersimpan
│   │
│   ├── 📂 images/
│   │   ├── 🖼️  anomaly_distribution.png
│   │   ├── 🖼️  anomaly_score_distribution.png
│   │   ├── 🖼️  boxplot_features.png
│   │   ├── 🖼️  correlation_heatmap.png
│   │   ├── 🖼️  feature_importance.png
│   │   ├── 🖼️  histogram_features.png
│   │   ├── 🖼️  pca_2d_anomaly.png
│   │   ├── 🖼️  pca_3d_anomaly.png
│   │   └── 🖼️  precision_recall_curve.png
│   │
│   └── 📂 reports/
│       └── 📄 anomaly_result.csv        ← Hasil lengkap deteksi anomali
│
├── 📑 laporan isolation forest.pdf      ← Laporan lengkap penelitian
└── 📊 Presentasi Skripsi - Deteksi Anomali Kartu Kredit (Isolation Forest).pptx
```

---

## 📊 Dataset

Dataset yang digunakan adalah **Credit Card Fraud Detection Dataset** — dataset publik yang banyak dipakai di dunia riset keamanan finansial.

| Atribut | Detail |
|--------|--------|
| 📁 File | `creditcard.csv` |
| 📏 Ukuran | ±144 MB |
| 🔢 Jumlah Baris | **284,807 transaksi** |
| 🧮 Jumlah Kolom | 31 kolom |
| ⚖️ Proporsi Fraud | ~0.17% (sangat imbalanced) |

### 🔑 Fitur Dataset

| Kolom | Keterangan |
|-------|-----------|
| `Time` | Detik sejak transaksi pertama |
| `V1` – `V28` | 28 fitur hasil transformasi **PCA** (anonim untuk privasi) |
| `Amount` | Nominal transaksi dalam USD |
| `Class` | Label: `0` = Normal, `1` = Fraud |

> ⚠️ **Catatan:** Kolom `V1`–`V28` telah dianonimkan menggunakan PCA oleh penyedia data demi menjaga kerahasiaan informasi nasabah.

---

## 🔬 Metodologi

Proyek ini mengikuti alur standar Machine Learning:

```
[📥 Load Dataset] → [🔍 EDA & Visualisasi] → [⚙️ Preprocessing]
       ↓
[🌲 Training Isolation Forest] → [📊 Prediksi & Scoring]
       ↓
[📈 Evaluasi Model] → [💾 Simpan Model & Hasil] → [🧪 Testing]
```

### Tahapan Detail:

**1️⃣ EDA (Exploratory Data Analysis)**
- Distribusi kelas (Normal vs Fraud)
- Statistik deskriptif fitur
- Korelasi antar fitur (heatmap)
- Distribusi nilai transaksi (Amount)

**2️⃣ Preprocessing**
- Normalisasi kolom `Amount` menggunakan `StandardScaler`
- Seleksi fitur: `V1`–`V28` + `Amount` (29 fitur total)

**3️⃣ Model Training**
- Melatih Isolation Forest pada seluruh dataset
- Menyimpan model dan scaler dalam format `.pkl`

**4️⃣ Prediksi & Analisis**
- Menghasilkan `Anomaly Score` untuk setiap transaksi
- Label: `1` = Normal, `-1` = Anomaly (Fraud)
- Analisis distribusi skor anomali

**5️⃣ Evaluasi**
- Precision-Recall Curve
- Confusion Matrix
- F1-Score, Accuracy, Recall

---

## 📈 Visualisasi Hasil

Berikut adalah visualisasi yang dihasilkan dari analisis model (tersimpan di `output/images/`):

| # | File | Deskripsi |
|---|------|-----------|
| 1 | `anomaly_distribution.png` | Persebaran data normal vs anomali |
| 2 | `anomaly_score_distribution.png` | Histogram distribusi anomaly score |
| 3 | `boxplot_features.png` | Boxplot distribusi 29 fitur utama |
| 4 | `correlation_heatmap.png` | Heatmap korelasi antar fitur |
| 5 | `feature_importance.png` | Kontribusi/kepentingan tiap fitur |
| 6 | `histogram_features.png` | Histogram distribusi tiap fitur |
| 7 | `pca_2d_anomaly.png` | Proyeksi PCA 2D: normal vs anomali |
| 8 | `pca_3d_anomaly.png` | Proyeksi PCA 3D interaktif |
| 9 | `precision_recall_curve.png` | Kurva Precision-Recall model |

---

## ⚙️ Cara Menjalankan

### 📋 Prasyarat

Pastikan Python **3.8+** sudah terinstal. Install semua dependensi:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter plotly
```

---

### 🔄 Langkah 1: Jalankan Notebook Utama

```bash
jupyter notebook isolation_forest_anomaly.ipynb
```

Jalankan semua cell dari atas ke bawah. Notebook ini akan:
- Melakukan EDA lengkap
- Melatih model Isolation Forest
- Menyimpan model ke `output/model/`
- Menghasilkan semua visualisasi ke `output/images/`
- Menyimpan hasil prediksi ke `output/reports/`

---

### 🧪 Langkah 2: Uji Model dengan Data Baru

```bash
python test_model.py
```

**Contoh output:**

```
════════════════════════════════════════════════════════════
  1. LOAD MODEL & SCALER
════════════════════════════════════════════════════════════
  ✔  Model  loaded : output/model/isolation_forest.pkl
  ✔  Scaler loaded : output/model/standard_scaler.pkl
  ✔  n_estimators  : 100
  ✔  contamination : auto

════════════════════════════════════════════════════════════
  4. PREDIKSI MODEL
════════════════════════════════════════════════════════════
  Total data       : xxx
  Prediksi Normal  : xxx
  Prediksi Anomaly : xxx  (x.x%)

════════════════════════════════════════════════════════════
  5. DETAIL HASIL
════════════════════════════════════════════════════════════
  🚨  [TXN-001]  Amount=    999.00  Score= -0.1234  ← MENCURIGAKAN   Pred=ANOMALY
  ✓   [TXN-002]  Amount=      5.50  Score=  0.0567  ← aman           Pred=Normal
```

---

### 📊 Langkah 3: Cek Hasil

| File Output | Keterangan |
|------------|-----------|
| `test_result.csv` | Hasil prediksi lengkap pada `test_data.csv` |
| `output/reports/anomaly_result.csv` | Hasil prediksi pada seluruh dataset |
| `output/images/*.png` | Semua visualisasi hasil analisis |
| `output/model/*.pkl` | Model & scaler yang sudah terlatih |

---

## 📦 Dependensi

| Library | Versi Minimum | Fungsi |
|---------|--------------|--------|
| `pandas` | ≥ 1.3.0 | Manipulasi dan analisis data |
| `numpy` | ≥ 1.21.0 | Komputasi numerik |
| `scikit-learn` | ≥ 0.24.0 | Isolation Forest, Scaler, Metrics |
| `matplotlib` | ≥ 3.4.0 | Visualisasi grafik dasar |
| `seaborn` | ≥ 0.11.0 | Visualisasi statistik lanjutan |
| `joblib` | ≥ 1.0.0 | Simpan dan load model .pkl |
| `jupyter` | ≥ 1.0.0 | Notebook interaktif |
| `plotly` | ≥ 5.0.0 | Visualisasi 3D interaktif |

---

## 📉 Hasil & Evaluasi Model

### 🎯 Metrik Evaluasi

| Metrik | Keterangan |
|--------|-----------|
| **Accuracy** | % transaksi yang diklasifikasikan dengan benar |
| **Precision** | Dari yang diprediksi Fraud, berapa % yang benar-benar Fraud |
| **Recall** | Dari seluruh Fraud nyata, berapa % yang berhasil terdeteksi |
| **F1-Score** | Harmonic mean antara Precision dan Recall |

### 📐 Confusion Matrix Konsep

```
                      PREDIKSI
                  Normal    ANOMALY
AKTUAL  Normal  [  TN   ] [  FP   ]
        Fraud   [  FN   ] [  TP   ]

TN = True Negative  → Normal diprediksi Normal  ✅
TP = True Positive  → Fraud  diprediksi Fraud   ✅
FP = False Positive → Normal diprediksi Fraud   ❌ (False Alarm)
FN = False Negative → Fraud  diprediksi Normal  ❌ (Missed Fraud!)
```

> 💡 Dalam fraud detection, **Recall (sensitivity)** adalah metrik paling kritis — kita tidak boleh melewatkan satu pun transaksi fraud!

---

## 🔍 Cara Kerja `test_model.py`

Script pengujian berjalan dalam **7 tahap otomatis:**

```
Step 1 │ Load Model & Scaler      → Memuat .pkl dari output/model/
Step 2 │ Load Test Data           → Membaca test_data.csv + validasi kolom
Step 3 │ Preprocessing (Scaling)  → Normalisasi dengan StandardScaler
Step 4 │ Prediksi                 → Menjalankan model, output score & label
Step 5 │ Detail Hasil             → Menampilkan tiap transaksi (🚨 atau ✓)
Step 6 │ Ringkasan Akurasi        → Hitung TP/TN/FP/FN + Precision/Recall/F1
Step 7 │ Simpan Hasil             → Export ke test_result.csv
```

---

## 📄 Laporan & Presentasi

| Dokumen | Keterangan |
|--------|-----------|
| 📑 `laporan isolation forest.pdf` | Laporan penelitian lengkap dengan metodologi, analisis, dan kesimpulan |
| 📊 `Presentasi Skripsi - Deteksi Anomali Kartu Kredit (Isolation Forest).pptx` | Slide presentasi skripsi/tugas akhir |

---

## 📌 Catatan Teknis

### ⚠️ Penting Sebelum Menjalankan

1. **Jalankan notebook terlebih dahulu** — file model `.pkl` harus sudah ada di `output/model/` sebelum menjalankan `test_model.py`
2. **File `creditcard.csv` berukuran ±144 MB** — proses loading dan training memerlukan beberapa menit tergantung spesifikasi komputer
3. **Format `test_data.csv`** harus memiliki kolom `V1`–`V28` dan `Amount` minimal

### 📂 Format Data Uji (`test_data.csv`)

| Kolom | Wajib | Keterangan |
|-------|-------|-----------|
| `V1` – `V28` | ✅ Ya | 28 fitur PCA utama |
| `Amount` | ✅ Ya | Nilai nominal transaksi |
| `TrueLabel` | ❌ Opsional | Label asli (untuk kalkulasi akurasi) |
| `TransactionID` | ❌ Opsional | ID unik transaksi |

**Contoh baris `test_data.csv`:**

```csv
V1,V2,...,V28,Amount,TrueLabel,TransactionID
-1.35,-0.07,...,-0.02,149.62,Fraud_Clear,TXN-001
1.19,0.26,...,0.08,2.69,Normal,TXN-002
```

---

<div align="center">

---

### 🙏 Terima Kasih

**Proyek ini dibuat sebagai bagian dari penelitian di bidang Machine Learning dan Keamanan Finansial.**

> *"Isolation Forest membuktikan bahwa anomali bukan sekadar 'berbeda' —*  
> *ia lebih mudah ditemukan justru karena keunikannya sendiri."*

<br>

**Muhammad Sahal Anwar Hadi** · NIM: **24260032**

<br>

![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Powered%20by-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

</div>
