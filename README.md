# Praktikum-LAB

# Aplikasi Prediksi Polusi (Regresi)

Aplikasi ini dibuat menggunakan Streamlit untuk melakukan prediksi tingkat polusi udara menggunakan beberapa algoritma regresi seperti Linear Regression, Support Vector Regression (SVR), dan Neural Network (MLPRegressor). Anda dapat melakukan preprocessing data, memilih target prediksi, melakukan scaling fitur, dan membandingkan performa model secara interaktif.

## Fitur Utama

- **Upload Dataset CSV**: Unggah data polusi udara Anda dalam format CSV.
- **Preprocessing Data**:
  - Penanganan missing values (Mean, Median, Mode, Hapus Baris)
  - Penanganan duplikasi data
  - Feature engineering (ekstraksi fitur waktu dari kolom datetime)
  - One-hot encoding untuk kolom kategorikal
- **Pemilihan Target & Fitur**: Pilih kolom target (PM10, SO2, NO2, CO, O3, TEMP, RAIN, WSPM) dan fitur numerik.
- **Scaling Fitur**: Opsi untuk melakukan standardisasi fitur numerik.
- **Pilihan Algoritma Regresi**:
  - Linear Regression
  - Support Vector Regression (SVR) dengan pengaturan hyperparameter
  - Neural Network (MLPRegressor) dengan pengaturan arsitektur dan hyperparameter
- **Evaluasi Model**: Menampilkan metrik MAE, MSE, RMSE, R2 Score, serta visualisasi Prediksi vs Aktual dan Residuals Plot.

## Cara Menjalankan

1. **Install dependencies**  
   Pastikan Anda sudah menginstall Streamlit dan library yang dibutuhkan:

   ```sh
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Jalankan aplikasi**

   ```sh
   streamlit run app.py
   ```

3. **Buka browser**  
   Akses aplikasi melalui link yang muncul di terminal (biasanya http://localhost:8501).

## Struktur Folder

```
app.py
tutorial.py
Tes-data.ipynb
data/
    dataset_numerik_manual.csv
    Tes_data_clean.csv
```

- **app.py**: Source utama aplikasi Streamlit.
- **tutorial.py**: Panduan interaktif penggunaan aplikasi.
- **Tes-data.ipynb**: Notebook untuk eksplorasi dan preprocessing data.
- **data/**: Folder untuk dataset.

## Panduan Penggunaan

1. **Unggah Dataset**

   - Klik "Pilih file CSV" dan upload dataset Anda.

2. **Preprocessing**

   - Pilih strategi penanganan missing values dan duplikasi di sidebar.
   - Lakukan ekstraksi fitur waktu jika ada kolom datetime.
   - Terapkan one-hot encoding jika ada kolom kategorikal.

3. **Pilih Target & Fitur**

   - Pilih kolom target polusi yang ingin diprediksi.
   - Pilih fitur numerik yang digunakan untuk prediksi.

4. **Scaling & Split Data**

   - Aktifkan scaling fitur jika menggunakan SVR/Neural Network.
   - Atur proporsi data uji.

5. **Pilih Algoritma & Jalankan**

   - Pilih algoritma regresi dan atur hyperparameter jika perlu.
   - Klik "Jalankan Prediksi Polusi".

6. **Lihat Hasil**
   - Evaluasi performa model melalui metrik dan visualisasi yang ditampilkan.

## Catatan

- Untuk hasil optimal, pastikan data sudah bersih dari missing values dan duplikasi.
- Scaling fitur sangat penting untuk SVR dan Neural Network.
- Lihat file [tutorial.py](tutorial.py) untuk panduan interaktif lebih lanjut.

---

© 2025 Praktikum LAB - Prediksi Polusi Udara
