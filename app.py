import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # For regression
from sklearn.svm import SVR # Import Support Vector Regressor
from sklearn.neural_network import MLPRegressor # Import Multi-layer Perceptron Regressor (Neural Network)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Regression metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Aplikasi Prediksi Polusi (Regresi)")
st.title("Aplikasi Prediksi Polusi (Regresi)")
st.write("Unggah dataset Anda (CSV) pilih tahun, lakukan preprocessing, dan prediksi tingkat polusi.")

# --- Bagian Upload File ---
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
df_original = None  # DataFrame asli setelah diunggah
df_processed = None # DataFrame setelah preprocessing

if uploaded_file is not None:
    try:
        df_original = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil diunggah!")
        st.subheader("Preview Data Awal:")
        st.dataframe(df_original.head())

        st.subheader("Statistik Deskriptif Awal:")
        st.dataframe(df_original.describe())

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        df_original = None

# --- Bagian Preprocessing Data ---
if df_original is not None:
    st.sidebar.header("Pengaturan Preprocessing Data")

    # 1. Penanganan Missing Values
    st.sidebar.subheader("1. Penanganan Missing Values")
    missing_info = df_original.isnull().sum()
    missing_cols = missing_info[missing_info > 0]

    if not missing_cols.empty:
        st.sidebar.write("Kolom dengan Missing Values:")
        st.sidebar.dataframe(missing_cols)

        # Untuk Kolom Numerik
        numeric_missing_cols = missing_cols[df_original[missing_cols.index].dtypes != 'object'].index.tolist()
        numeric_imputation_strategy = None # Initialize
        if numeric_missing_cols:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Strategi untuk Missing Numerik:**")
            numeric_imputation_strategy = st.sidebar.selectbox(
                "Pilih strategi:",
                ("Hapus Baris", "Isi dengan Mean", "Isi dengan Median"),
                key="num_impute_strategy"
            )
            if numeric_imputation_strategy == "Isi dengan Mean":
                for col in numeric_missing_cols:
                    df_original[col].fillna(df_original[col].mean(), inplace=True)
            elif numeric_imputation_strategy == "Isi dengan Median":
                for col in numeric_missing_cols:
                    df_original[col].fillna(df_original[col].median(), inplace=True)

        # Untuk Kolom Kategorikal (Object)
        categorical_missing_cols = missing_cols[df_original[missing_cols.index].dtypes == 'object'].index.tolist()
        categorical_imputation_strategy = None # Initialize
        if categorical_missing_cols:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Strategi untuk Missing Kategorikal (Object):**")
            categorical_imputation_strategy = st.sidebar.selectbox(
                "Pilih strategi:",
                ("Hapus Baris", "Isi dengan Mode", "Isi dengan 'Missing'"),
                key="cat_impute_strategy"
            )
            if categorical_imputation_strategy == "Isi dengan Mode":
                for col in categorical_missing_cols:
                    df_original[col].fillna(df_original[col].mode()[0], inplace=True)
            elif categorical_imputation_strategy == "Isi dengan 'Missing'":
                for col in categorical_missing_cols:
                    df_original[col].fillna('Missing', inplace=True)

        # Terapkan hapus baris jika dipilih di salah satu strategi
        if "Hapus Baris" in [numeric_imputation_strategy, categorical_imputation_strategy] and (numeric_missing_cols or categorical_missing_cols):
            initial_rows_after_impute = len(df_original)
            df_original.dropna(inplace=True)
            dropped_rows_impute = initial_rows_after_impute - len(df_original)
            if dropped_rows_impute > 0:
                st.sidebar.warning(f"{dropped_rows_impute} baris dengan missing values telah dihapus.")
            else:
                st.sidebar.info("Tidak ada baris yang dihapus karena missing values setelah imputasi.")
    else:
        st.sidebar.info("Tidak ada missing values terdeteksi.")


    # 2. Penanganan Duplikasi
    st.sidebar.subheader("2. Penanganan Duplikasi")
    if st.sidebar.checkbox("Hapus baris duplikat?", value=True):
        initial_rows = len(df_original)
        df_original.drop_duplicates(inplace=True)
        dropped_rows = initial_rows - len(df_original)
        if dropped_rows > 0:
            st.sidebar.success(f"{dropped_rows} baris duplikat berhasil dihapus.")
        else:
            st.sidebar.info("Tidak ada baris duplikat ditemukan.")

    df_processed = df_original.copy() # Mulai df_processed dari df_original yang sudah ditangani MV/Duplikasi

    # 3. Feature Engineering & Encoding (datetime, wd, station)
    st.sidebar.subheader("3. Feature Engineering & Encoding")

    # Datetime Feature Engineering
    if 'datetime' in df_processed.columns:
        if st.sidebar.checkbox("Ekstrak fitur dari 'datetime' (year, month, day, hour, dayofweek)?", value=True):
            try:
                df_processed['datetime'] = pd.to_datetime(df_processed['datetime'], errors='coerce')
                # Hapus baris jika konversi datetime gagal untuk menghindari NaN di fitur waktu
                initial_rows_datetime = len(df_processed)
                df_processed.dropna(subset=['datetime'], inplace=True)
                dropped_rows_datetime = initial_rows_datetime - len(df_processed)
                if dropped_rows_datetime > 0:
                    st.sidebar.warning(f"{dropped_rows_datetime} baris dihapus karena format datetime tidak valid.")

                if not df_processed.empty:
                    df_processed['year'] = df_processed['datetime'].dt.year
                    df_processed['month'] = df_processed['datetime'].dt.month
                    df_processed['day'] = df_processed['datetime'].dt.day
                    df_processed['hour'] = df_processed['datetime'].dt.hour
                    df_processed['dayofweek'] = df_processed['datetime'].dt.dayofweek
                    df_processed.drop(columns=['datetime'], inplace=True)
                    st.sidebar.success("Fitur 'year', 'month', 'day', 'hour', 'dayofweek' berhasil diekstrak.")
                else:
                    st.sidebar.error("Dataset kosong setelah menghapus baris datetime tidak valid.")
            except Exception as e:
                st.sidebar.error(f"Gagal mengekstrak fitur datetime: {e}. Pastikan format benar.")
        else:
            if 'datetime' in df_processed.columns: # Jika ada tapi tidak diekstrak, hapus agar tidak error di ML
                 df_processed.drop(columns=['datetime'], inplace=True)


    # One-Hot Encoding untuk Kolom Kategorikal (selain datetime)
    categorical_cols_for_ohe = [col for col in df_processed.select_dtypes(include='object').columns]
    if categorical_cols_for_ohe:
        st.sidebar.write(f"Kolom kategorikal terdeteksi: {', '.join(categorical_cols_for_ohe)}")
        if st.sidebar.checkbox("Lakukan One-Hot Encoding untuk kolom kategorikal?", value=True):
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols_for_ohe, drop_first=True) # drop_first=True
            st.sidebar.success("One-Hot Encoding berhasil diterapkan.")
    else:
        st.sidebar.info("Tidak ada kolom kategorikal yang tersisa untuk One-Hot Encoding.")


    st.subheader("Informasi Data Setelah Preprocessing:")
    buffer_processed = io.StringIO()
    df_processed.info(buf=buffer_processed)
    s_processed = buffer_processed.getvalue()
    st.text(s_processed)

    st.write("**Jumlah Missing Values per Kolom:**")
    st.write(df_processed.isna().sum())

    st.write(f"**Jumlah Baris Duplikat:** {df_processed.duplicated().sum()}")

    st.subheader("Statistik Deskriptif Setelah Preprocessing:")
    st.dataframe(df_processed.describe())

# --- Bagian Prediksi Polusi (Regresi) ---
if df_processed is not None and not df_processed.empty:
    st.sidebar.header("Pengaturan Prediksi Polusi (Regresi)")

    # Filter hanya kolom numerik untuk target regresi
    all_numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()

    if not all_numeric_cols:
        st.error("Tidak ada kolom numerik yang tersedia setelah preprocessing untuk Regresi.")
        st.stop()

    # Batasi hanya kolom yang relevan untuk target polusi/cuaca jika ada
    allowed_targets = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'RAIN', 'WSPM']
    available_targets = [col for col in allowed_targets if col in all_numeric_cols]

    if not available_targets:
        st.error("Kolom target polutan (PM10, SO2, NO2, CO, O3, TEMP, RAIN, WSPM) tidak ditemukan di dataset setelah preprocessing.")
        st.stop()

    # Pilih kolom target untuk regresi
    target_col = st.sidebar.selectbox(
        "Pilih Kolom Target (nilai numerik kontinu yang akan diprediksi):", available_targets
    )

    # Identifikasi fitur (semua kolom numerik kecuali target)
    feature_cols = [col for col in all_numeric_cols if col != target_col]

    if not feature_cols:
        st.warning("Pilih setidaknya satu kolom fitur numerik untuk memprediksi target.")
    else:
        X = df_processed[feature_cols]
        y = df_processed[target_col]

        st.sidebar.subheader("4. Scaling Fitur Numerik (Untuk Fitur Saja)")
        # Scaling fitur dilakukan di sini, setelah target dipisahkan
        use_scaling = st.sidebar.checkbox("Lakukan Feature Scaling (StandardScaler) pada Fitur?", value=True, key="regression_scaling")
        if use_scaling:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            st.sidebar.success("Feature Scaling berhasil diterapkan pada fitur (StandardScaler).")

        test_size = st.sidebar.slider("Ukuran Data Uji (%):", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.sidebar.write(f"Data latih: {len(X_train)} sampel, Data uji: {len(X_test)} sampel")

        st.sidebar.subheader("Pilih Algoritma Regresi")
        regressor_name = st.sidebar.selectbox("Pilih Regressor:", (
            "Linear Regression",
            "Support Vector Regression (SVR)",
            "Neural Network (MLPRegressor)"
        ))

        model = None
        if regressor_name == "Linear Regression":
            model = LinearRegression()
        elif regressor_name == "Support Vector Regression (SVR)":
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Hyperparameter SVR:**")
            svr_c = st.sidebar.slider("C (Regularization):", 0.1, 100.0, 1.0)
            svr_epsilon = st.sidebar.slider("Epsilon (Margin of Tolerance):", 0.01, 1.0, 0.1)
            svr_kernel = st.sidebar.selectbox("Kernel:", ("rbf", "linear", "poly"))
            model = SVR(C=svr_c, epsilon=svr_epsilon, kernel=svr_kernel)
            st.info("Catatan: SVR sangat sensitif terhadap scaling fitur. Pastikan 'Lakukan Feature Scaling' dicentang.")
        elif regressor_name == "Neural Network (MLPRegressor)":
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Hyperparameter Neural Network:**")
            nn_hidden_layers_options = {
                "1 Lapisan (50 neuron)": (50,),
                "1 Lapisan (100 neuron)": (100,),
                "2 Lapisan (50, 25 neuron)": (50, 25),
                "3 Lapisan (100, 50, 25 neuron)": (100, 50, 25)
            }
            nn_hidden_layers_key = st.sidebar.selectbox("Ukuran Hidden Layer:", list(nn_hidden_layers_options.keys()))
            nn_hidden_layers = nn_hidden_layers_options[nn_hidden_layers_key]

            nn_activation = st.sidebar.selectbox("Fungsi Aktivasi:", ("relu", "tanh", "logistic"))
            nn_solver = st.sidebar.selectbox("Solver Optimasi:", ("adam", "sgd"))
            nn_max_iter = st.sidebar.slider("Max Iterations (Pelatihan):", 100, 2000, 500)
            model = MLPRegressor(hidden_layer_sizes=nn_hidden_layers, activation=nn_activation,
                                 solver=nn_solver, max_iter=nn_max_iter, random_state=42)
            st.info("Catatan: Neural Network sangat sensitif terhadap scaling fitur. Pastikan 'Lakukan Feature Scaling' dicentang.")


        if model and st.sidebar.button("Jalankan Prediksi Polusi"):
            st.subheader(f"Hasil {regressor_name} untuk Prediksi {target_col}")
            try:
                # Periksa apakah X_train atau y_train kosong
                if X_train.empty or y_train.empty:
                    st.error("Data latih kosong. Pastikan dataset Anda memiliki cukup data valid setelah preprocessing.")
                    st.stop()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # --- Metrik Evaluasi Regresi ---
                st.markdown("### Metrik Evaluasi Model")

                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
                st.info(f"MAE adalah rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi. Semakin kecil nilai MAE ({mae:.2f}), semakin dekat prediksi Anda dengan nilai sebenarnya.")

                mse = mean_squared_error(y_test, y_pred)
                st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
                st.info(f"MSE adalah rata-rata dari kuadrat selisih antara nilai aktual dan prediksi. Ini memberikan bobot lebih besar pada kesalahan yang besar. Semakin kecil nilainya ({mse:.2f}), semakin baik.")

                rmse = np.sqrt(mse)
                st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.2f}")
                st.info(f"RMSE adalah akar kuadrat dari MSE, yang mengembalikan unit kesalahan ke unit yang sama dengan variabel target ({target_col}). Ini lebih mudah diinterpretasikan daripada MSE. Semakin kecil nilai RMSE ({rmse:.2f}), semakin akurat model Anda.")

                r2 = r2_score(y_test, y_pred)
                st.write(f"**R2 Score (Koefisien Determinasi):** {r2:.2f}")
                st.info(f"R2 Score mengukur seberapa baik model Anda menjelaskan varians dalam variabel target. Nilai berkisar antara 0 hingga 1. Nilai {r2:.2f} berarti model Anda dapat menjelaskan {r2*100:.0f}% dari variabilitas pada {target_col}. Semakin tinggi nilai R2 (mendekati 1.0), semakin baik model Anda dalam menjelaskan data.")

                # --- Visualisasi ---
                st.markdown("### Visualisasi Hasil Prediksi")

                st.subheader("Prediksi vs Nilai Aktual")
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.scatter(y_test, y_pred, alpha=0.5)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prediksi Ideal (Aktual = Prediksi)')
                ax.set_xlabel(f"Nilai Aktual {target_col}")
                ax.set_ylabel(f"Nilai Prediksi {target_col}")
                ax.set_title(f"Prediksi vs Nilai Aktual untuk {target_col}")
                ax.legend()
                st.pyplot(fig)
                st.markdown("""
                **Interpretasi Plot Prediksi vs Aktual:**
                - Setiap titik mewakili satu observasi. Sumbu X adalah nilai sebenarnya, sumbu Y adalah nilai prediksi model.
                - Garis putus-putus merah menunjukkan di mana prediksi ideal berada (prediksi sama dengan aktual).
                - **Jika titik-titik data berkumpul rapat di sekitar garis merah**, ini menunjukkan bahwa model Anda membuat prediksi yang sangat akurat.
                - **Jika titik-titik tersebar jauh dari garis**, model kurang akurat.
                - Pola tertentu (misalnya, berbentuk kipas atau melengkung) bisa mengindikasikan masalah dalam model.
                """)


                st.subheader("Residuals Plot")
                residuals = y_test - y_pred
                fig_res, ax_res = plt.subplots(figsize=(10, 7))
                ax_res.scatter(y_pred, residuals, alpha=0.5)
                ax_res.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
                ax_res.set_xlabel(f"Nilai Prediksi {target_col}")
                ax_res.set_ylabel("Residual (Aktual - Prediksi)")
                ax_res.set_title(f"Residuals Plot untuk {target_col}")
                st.pyplot(fig_res)
                st.markdown("""
                **Interpretasi Residuals Plot:**
                - Sumbu X adalah nilai yang diprediksi oleh model, dan sumbu Y adalah selisih antara nilai aktual dan prediksi (disebut *residual*).
                - Garis putus-putus merah berada di nol, yang berarti prediksi model sama persis dengan aktual.
                - **Idealnya, titik-titik *residual* harus tersebar secara acak di sekitar garis nol tanpa pola yang jelas**. Ini menunjukkan bahwa model telah menangkap sebagian besar pola dalam data dan tidak ada bias sistematis.
                - **Jika melihat pola** (misalnya, bentuk kerucut, kurva, atau pengelompokan), ini bisa menunjukkan bahwa model memiliki bias atau bahwa ada informasi penting yang tidak ditangkap oleh model (misalnya, fitur yang hilang atau hubungan non-linier yang tidak dimodelkan).
                """)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih atau mengevaluasi model: {e}")

else:
    if uploaded_file is None:
        st.info("Silakan unggah file CSV pilih tahun, untuk memulai prediksi polusi.")
    elif df_original is not None and df_original.empty:
        st.error("Dataset kosong setelah preprocessing. Periksa file CSV Anda.")