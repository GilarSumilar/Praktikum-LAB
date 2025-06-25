import streamlit as st

def run_tutorial():
    st.set_page_config(layout="wide")
    st.title("Panduan Penggunaan Aplikasi Prediksi Polusi (Regresi)")

    st.write(
        """
        Untuk menggunakan ketiga algoritma regresi yang baru saja ditambahkan (Linear Regression, Support Vector Regression/SVR, 
        dan Neural Network/MLPRegressor) di aplikasi Streamlit Anda, ikuti langkah-langkah interaktif di sidebar aplikasi. Untuk kasus ini, 
        Mengingat sifat data polutan yang kontinu dan tujuan untuk memprediksi tingkat atau nilai dari polutan tersebut, Linear Regression
        adalah algotirma yang saya sarankan.
        Gunakan PM10, SO2, NO2, CO, O3, sebagai kolom target untuk prediksi polusi udara. 
        """ 
    )

    st.header("Informasi yang Perlu Diketahui")

    st.write(
        """
        - year, month, day, hour: Menunjukkan tanggal dan waktu pengambilan data.
        - PM10: Partikel debu dengan diameter kurang dari 10 mikrometer. Semakin tinggi nilai, semakin buruk kualitas udara.
        - SO2, NO2, CO, O3: Gas polutan, yaitu Sulfur dioksida, Nitrogen dioksida, Karbon monoksida, dan Ozon. Semakin tinggi nilai, semakin buruk kualitas udara.
        - TEMP: Suhu udara.
        - RAIN: Curah hujan.
        - WSPM: Kecepatan angin.
        """
    )

    st.markdown("---")

    st.header("Langkah-langkah Umum (Untuk Semua Algoritma)")

    st.subheader("1. Unggah Dataset Anda")
    st.write(
        """
        - Buka aplikasi Streamlit di browser (setelah menjalankan `streamlit run app.py`).
        - Klik tombol **"Pilih file CSV, sesuai tahun yang di inginkan"** dan unggah dataset polusi Anda.
        """
    )

    st.subheader("2. Lakukan Preprocessing Data (di Sidebar Kiri)")
    st.write(
        """
        - Di bagian **"Pengaturan Preprocessing Data"**:
            - **Penanganan Missing Values**: Pilih strategi yang sesuai (Mean, Median, Mode, atau Hapus Baris) jika ada data yang hilang.
            - **Penanganan Duplikasi**: Centang **"Hapus baris duplikat?"** (disarankan).
            - **Feature Engineering & Encoding**:
                - Jika dataset Anda memiliki kolom datetime, centang **"Ekstrak fitur dari datetime"** (disarankan untuk data deret waktu).
                - Jika ada kolom kategorikal lain (seperti `wd` atau `station`), centang **"Lakukan One-Hot Encoding"**.
        - Setelah semua langkah preprocessing, Anda akan melihat **"Preview Data Setelah Preprocessing"** dan **"Informasi Data Setelah Preprocessing"** di bagian utama aplikasi.
        """
    )

    st.subheader("3. Pilih Kolom Target & Fitur (di Sidebar Kiri)")
    st.write(
        """
        - Di bagian **"Pengaturan Prediksi Polusi (Regresi)"**:
            - **Pilih Kolom Target**: Pilih kolom polutan yang ingin diprediksi (misal PM10, SO2, NO2, CO, O3).
            - Secara otomatis, semua kolom numerik lain yang bukan target akan digunakan sebagai Kolom Fitur. Anda bisa menyesuaikannya di multiselect **"Pilih Kolom Fitur"** jika ingin memilih subset tertentu.
            - **Penting: Scaling Fitur Numerik**: Pastikan kotak **"Lakukan Feature Scaling (StandardScaler) pada Fitur?"** dicentang. Ini sangat penting untuk SVR dan Neural Network agar bekerja dengan baik.
        """
    )

    st.subheader("4. Atur Ukuran Data Uji")
    st.write(
        """
        - Sesuaikan slider **"Ukuran Data Uji (%)"** (misal 0.20 untuk 20% data uji).
        """
    )

    st.markdown("---")

    st.header("Cara Menggunakan Masing-masing Algoritma")

    st.write(
        """
        Sekarang, di bagian **"Pilih Algoritma Regresi"**, Anda bisa memilih satu per satu algoritma yang ingin dicoba:
        """
    )

    st.subheader("1. Linear Regression")
    st.write(
        """
        - **Langkah**:
            - Di dropdown **"Pilih Regressor:"**, pilih **"Linear Regression"**.
            - Tidak ada hyperparameter tambahan yang perlu diatur.
        - **Jalankan**:
            - Klik tombol **"Jalankan Prediksi Polusi"**.
        - **Output**: Aplikasi akan menampilkan metrik (MAE, MSE, RMSE, R2 Score) dan plot (Prediksi vs Aktual, Residuals Plot) beserta penjelasannya.
        """
    )

    st.subheader("2. Support Vector Regression (SVR)")
    st.write(
        """
        - **Langkah**:
            - Di dropdown **"Pilih Regressor:"**, pilih **"Support Vector Regression (SVR)"**.
            - Akan muncul pengaturan hyperparameter SVR di sidebar:
                - **C (Regularization)**: Geser slider. Nilai lebih tinggi (misal 10.0 atau 100.0) membuat model lebih fokus pada fitting data pelatihan (berpotensi overfitting), nilai lebih rendah (misal 0.1) membuat model lebih umum.
                - **Epsilon (Margin of Tolerance)**: Geser slider. Menentukan margin toleransi tanpa penalti untuk kesalahan di dalam margin ini. Nilai kecil (misal 0.01) lebih ketat, nilai besar (misal 1.0) lebih longgar.
                - **Kernel**: Pilih jenis kernel (fungsi pemetaan data ke ruang berdimensi lebih tinggi).
                    - `rbf` (Radial Basis Function) adalah default yang umum dan baik untuk hubungan non-linear.
                    - `linear` untuk hubungan linear.
                    - `poly` (polynomial) untuk hubungan polinomial.
        - **Jalankan**:
            - Klik tombol **"Jalankan Prediksi Polusi"**.
        - **Output**: Aplikasi akan menampilkan metrik dan plot. Perhatikan perubahan performa dengan hyperparameter yang berbeda.
        """
    )

    st.subheader("3. Neural Network (MLPRegressor)")
    st.write(
        """
        - **Langkah**:
            - Di dropdown **"Pilih Regressor:"**, pilih **"Neural Network (MLPRegressor)"**.
            - Akan muncul pengaturan hyperparameter Neural Network di sidebar:
                - **Ukuran Hidden Layer**: Pilih arsitektur jaringan (misal `(50,)` untuk satu lapisan tersembunyi dengan 50 neuron, atau `(50, 25)` untuk dua lapisan dengan 50 dan 25 neuron).
                - **Fungsi Aktivasi**: Pilih fungsi aktivasi untuk neuron di lapisan tersembunyi.
                    - `relu` (Rectified Linear Unit) adalah yang paling umum.
                    - `tanh` (tangent hiperbolik)
                    - `logistic` (sigmoid, kurang umum untuk lapisan tersembunyi).
                - **Solver Optimasi**: Algoritma untuk mengoptimalkan bobot jaringan.
                    - `adam` adalah default yang umum.
                    - `sgd` (Stochastic Gradient Descent) adalah metode klasik.
                - **Max Iterations (Pelatihan)**: Geser slider. Jumlah maksimum iterasi pelatihan. Semakin tinggi, semakin lama pelatihan, tetapi bisa konvergensi lebih baik. Jika ada peringatan konvergensi, tingkatkan nilai ini.
        - **Jalankan**:
            - Klik tombol **"Jalankan Prediksi Polusi"**.
        - **Output**: Aplikasi akan menampilkan metrik dan plot. Neural Network bisa sangat kuat, tetapi juga rentan terhadap overfitting dan membutuhkan lebih banyak data serta penyesuaian hyperparameter yang cermat.
        """
    )

    st.markdown("---")

    st.header("Tips Saat Menggunakan 3 Algoritma Ini")
    st.write(
        """
        - **Bandingkan Hasil**: Setelah menjalankan setiap algoritma, catat metrik (terutama **RMSE** dan **R2 Score**) dan amati pola pada plot. Algoritma mana yang memberikan RMSE terkecil dan R2 Score tertinggi?
        - **Eksperimen dengan Hyperparameter**: Jangan ragu mengubah pengaturan hyperparameter untuk SVR dan Neural Network. Perubahan kecil bisa berdampak besar pada kinerja.
        - **Pentingnya Scaling**: Ingat, SVR dan Neural Network sangat bergantung pada feature scaling. Pastikan selalu mencentang opsi **"Lakukan Feature Scaling (StandardScaler) pada Fitur?"** di sidebar.
        - **Data**: Neural Network cenderung membutuhkan lebih banyak data untuk hasil optimal dibandingkan SVR atau Linear Regression.
        """
    )

    st.success("Dengan panduan ini, Anda bisa memanfaatkan kemampuan regresi yang berbeda di aplikasi Streamlit Anda!")

if __name__ == "__main__":
    run_tutorial()