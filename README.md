# Analisis Hubungan Tingkat Inflasi dan Skor Kebahagiaan Antarnegara Periode 2015–2023 Menggunakan Regresi

Repository proyek akhir mata kuliah **Data Science** — **Kelompok 24: Data Sunda**.

Proyek ini menganalisis hubungan antara tingkat inflasi dan skor kebahagiaan antarnegara selama periode 2015–2023 dengan pendekatan regresi linear. Analisis dilakukan melalui Jupyter Notebook, sedangkan hasil eksplorasi dan pemodelan disajikan dalam dashboard interaktif berbasis Streamlit.

---

## Anggota Kelompok

| No. | Nama | NIM |
|---:|---|---|
| 1 | Aulia Rahma Habibah | 123240190 |
| 2 | Sabrina Alya Fadhillah | 123240196 |
| 3 | Bisma Putra Pangestu | 123240217 |
| 4 | Muhammad Syawal A. | 123240224 |

---

## Latar Belakang

Inflasi merupakan salah satu indikator ekonomi yang dapat memengaruhi daya beli dan kondisi ekonomi masyarakat. Sementara itu, skor kebahagiaan digunakan sebagai salah satu gambaran kesejahteraan subjektif penduduk suatu negara.

Proyek ini berangkat dari ketertarikan untuk mengetahui apakah perubahan tingkat inflasi memiliki hubungan dengan skor kebahagiaan antarnegara. Analisis dilakukan pada data negara dan tahun yang tersedia dalam dataset periode 2015–2023.

**Penting:** analisis regresi dalam proyek ini digunakan untuk mengidentifikasi pola hubungan statistik. Hasilnya tidak secara otomatis membuktikan bahwa inflasi menyebabkan perubahan tingkat kebahagiaan.

## Tujuan Proyek

1. Memahami karakteristik data inflasi dan skor kebahagiaan antarnegara.
2. Melakukan eksplorasi dan persiapan data sebelum pemodelan.
3. Menganalisis hubungan antara inflasi utama (*Headline Consumer Price Inflation*) dan skor kebahagiaan (`Score`).
4. Membangun dan membandingkan model regresi linear sederhana dan regresi linear berganda.
5. Mengevaluasi performa model menggunakan metrik regresi.
6. Menyajikan hasil analisis melalui dashboard interaktif.

## Dataset dan Variabel

Dataset utama: `WHI_Inflation (1).csv`

Dataset menggabungkan informasi negara dan tahun dengan indikator inflasi serta sejumlah indikator yang berkaitan dengan skor kebahagiaan. Variabel yang digunakan dalam analisis utama meliputi:

| Variabel | Peran | Keterangan |
|---|---|---|
| `Country` | Identitas | Nama negara |
| `Year` | Identitas | Tahun pengamatan |
| `Headline Consumer Price Inflation` | Prediktor utama | Tingkat inflasi konsumen utama |
| `Energy Consumer Price Inflation` | Prediktor | Inflasi harga energi |
| `Food Consumer Price Inflation` | Prediktor | Inflasi harga makanan |
| `GDP deflator Index growth rate` | Prediktor | Pertumbuhan indeks GDP deflator |
| `Score` | Target | Skor kebahagiaan |

Dataset juga memuat variabel lain, seperti inflasi inti, inflasi produsen, GDP per kapita, dukungan sosial, harapan hidup sehat, kebebasan memilih, kemurahan hati, dan persepsi korupsi. Ketersediaan nilai dapat berbeda antarvariabel.

## Metode dan Alur Analisis

Notebook mengikuti alur kerja **CRISP-DM**:

1. **Business Understanding** — merumuskan masalah dan tujuan analisis.
2. **Data Understanding** — memahami struktur, tipe data, statistik deskriptif, nilai kosong, dan karakteristik data.
3. **Data Preparation** — menyiapkan data yang diperlukan untuk analisis dan pemodelan.
4. **Modeling** — membangun model regresi linear.
5. **Evaluation** — membandingkan hasil prediksi dan mengevaluasi performa model.
6. **Deployment** — menyajikan eksplorasi dan simulasi model melalui dashboard Streamlit.

### Model yang digunakan

- **Regresi Linear Sederhana**: menggunakan `Headline Consumer Price Inflation` sebagai prediktor untuk memperkirakan `Score`.
- **Regresi Linear Berganda**: menggunakan `Headline Consumer Price Inflation`, `Energy Consumer Price Inflation`, `Food Consumer Price Inflation`, dan `GDP deflator Index growth rate` sebagai prediktor untuk memperkirakan `Score`.
- **Baseline**: menggunakan rata-rata nilai target pada data training sebagai pembanding sederhana.

Pembagian data evaluasi menggunakan 80% data training dan 20% data testing dengan `random_state=42`. Perbandingan model dilakukan pada observasi lengkap yang sama agar hasil evaluasi dapat dibandingkan secara lebih adil.

### Metrik Evaluasi

- **R² (Coefficient of Determination)**: menunjukkan proporsi variasi target yang dapat dijelaskan model. Nilai negatif pada data uji berarti performa model lebih buruk daripada baseline rata-rata data training.
- **MAE (Mean Absolute Error)**: rata-rata besar kesalahan absolut prediksi.
- **RMSE (Root Mean Squared Error)**: ukuran kesalahan prediksi yang memberi penalti lebih besar pada kesalahan yang besar.

Notebook juga memuat pemeriksaan dan analisis tambahan, termasuk residual, heteroskedastisitas, multikolinearitas, robust standard errors, sensitivitas, serta validasi berdasarkan negara.

## Dashboard Streamlit

Dashboard menyediakan beberapa halaman/tab:

1. **Ringkasan** — indikator utama, tren tahunan, dan hubungan inflasi dengan skor kebahagiaan.
2. **Eksplorasi Data** — statistik deskriptif, nilai kosong, distribusi skor, dan observasi inflasi tertinggi.
3. **Pemodelan** — perbandingan baseline, regresi linear sederhana, dan regresi linear berganda menggunakan R², MAE, dan RMSE; termasuk grafik prediksi versus aktual.
4. **Simulasi Prediksi** — simulasi estimasi skor berdasarkan masukan nilai inflasi.
5. **Data** — melihat data sesuai filter dan mengunduh hasil filter dalam format CSV.

Filter tahun dan wilayah pada dashboard memengaruhi ringkasan serta grafik eksplorasi. Evaluasi model mengikuti rancangan pembagian data yang dijelaskan pada bagian metode. Simulasi prediksi merupakan demonstrasi model, bukan ramalan resmi.

## Struktur Repository

```text
DataSunda_Dashboard/
├── app.py
├── 24_IF_B_Data_Sunda_Progress4_Revisi_Diperbaiki.ipynb
├── WHI_Inflation (1).csv
├── requirements.txt
└── README.md
```

## Teknologi dan Library

- Python
- Jupyter Notebook
- Pandas dan NumPy
- Scikit-learn
- Statsmodels
- Matplotlib dan Seaborn
- Plotly
- Streamlit

## Instalasi dan Cara Menjalankan

### 1. Clone repository

```bash
git clone <URL-REPOSITORY-GITHUB>
cd <NAMA-FOLDER-REPOSITORY>
```

Ganti `<URL-REPOSITORY-GITHUB>` dengan URL repository milik kelompok.

### 2. Buat virtual environment (disarankan)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan dashboard

```bash
streamlit run app.py
```

Streamlit biasanya membuka dashboard pada alamat lokal:

```text
http://localhost:8501
```

Pastikan file CSV berada satu folder dengan `app.py`, atau unggah dataset melalui fitur *file uploader* pada sidebar dashboard.

### 5. Jalankan notebook

Buka file `24_IF_B_Data_Sunda_Progress4_Revisi_Diperbaiki.ipynb` melalui Jupyter Notebook, JupyterLab, atau VS Code. Pastikan dataset CSV tersedia sesuai lokasi yang digunakan notebook, kemudian jalankan sel secara berurutan.

## Keterbatasan

- Kelengkapan data tidak selalu sama pada setiap variabel dan negara.
- Hubungan inflasi dengan kebahagiaan dapat dipengaruhi banyak faktor lain yang tidak seluruhnya dimasukkan ke model.
- Regresi linear memiliki asumsi dan keterbatasan dalam menangkap hubungan yang kompleks.
- Performa pada data uji dan kemampuan generalisasi ke negara yang tidak muncul pada data training dapat berbeda.
- Hasil analisis menunjukkan hubungan statistik, bukan bukti sebab-akibat.
- Simulasi pada dashboard harus ditafsirkan dalam rentang dan konteks data yang digunakan.

## Kesimpulan Umum

Proyek ini menggunakan regresi linear untuk mempelajari hubungan antara tingkat inflasi dan skor kebahagiaan antarnegara. Model sederhana dan berganda dibandingkan dengan metrik evaluasi regresi, kemudian hasilnya disajikan melalui dashboard Streamlit agar lebih mudah dieksplorasi.

Hasil dan interpretasi akhir mengacu pada keluaran notebook serta evaluasi model. Performa prediksi yang terbatas tetap perlu disampaikan sebagai bagian dari temuan dan keterbatasan penelitian.

---

**Kelompok 24 — Data Sunda**  
Proyek Akhir Mata Kuliah Data Science
