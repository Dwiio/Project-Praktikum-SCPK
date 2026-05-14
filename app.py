import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SPK Pemilihan Motor Bekas - WP",
    page_icon="🏍️",
    layout="wide"
)

# --- 2. LOAD DATASET (Kaggle: Motorcycle Dataset) ---
@st.cache_data
def load_data():
    file_path = "BIKE DETAILS.csv"
    try:
        # Memuat dataset (Memenuhi syarat minimal 250 baris)
        df_raw = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "nehalbirla/motorcycle-dataset",
            file_path
        )
        return df_raw.drop_duplicates()
    except Exception as e:
        st.error(f"Gagal memuat dataset dari Kaggle: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. NAVIGASI SIDEBAR (Ketentuan 2a) ---
with st.sidebar:
    st.title("Menu Navigasi")
    # Pastikan opsi di sini sama persis dengan yang ada di struktur kontrol if/elif
    page = st.radio("Pilih Halaman:", ["Halaman Data", "Hitung SPK (WP)", "Halaman Profil Kelompok"])
    st.write("---")

# --- 4. HALAMAN 1: TAMPILAN DATASET (Ketentuan 2b) ---
if page == "Halaman Data":
    st.header("Dataset Mentah - Motorcycle Dataset")
    st.write(f"Menampilkan dataset interaktif dengan total **{len(df)}** baris data.")
    st.dataframe(df, use_container_width=True)

# --- 5. HALAMAN 2: PERHITUNGAN SPK WP (Ketentuan 2c, 2d, 2e, 5a) ---
elif page == "Hitung SPK (WP)":
    st.header("Sistem Pendukung Keputusan - Metode Weighted Product")
    
    # Preprocessing: Membuat 5 Kriteria agar memenuhi Ketentuan 4
    df_calc = df.copy()
    df_calc['usia_motor'] = 2026 - df_calc['year']
    df_calc['skor_kondisi'] = 100 - (df_calc['km_driven'] / 1500) - (df_calc['usia_motor'] * 1.5)
    df_calc['skor_kondisi'] = df_calc['skor_kondisi'].clip(lower=10)

    st.subheader("1. Input Bobot Kriteria (Dinamis)")
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.slider("Bobot Harga Jual (C1 - Cost)", 1, 5, 5)
        w2 = st.slider("Bobot Tahun Produksi (C2 - Benefit)", 1, 5, 4)
        w3 = st.slider("Bobot Jarak Tempuh/KM (C3 - Cost)", 1, 5, 3)
    with col2:
        w4 = st.number_input("Bobot Usia Kendaraan (C4 - Cost)", 1, 5, 2)
        w5 = st.number_input("Bobot Estimasi Kondisi (C5 - Benefit)", 1, 5, 4)

    if st.button("Hitung Peringkat Sekarang"):
        st.write("---")
        
        # PROSES WP
        weights = np.array([w1, w2, w3, w4, w5])
        total_w = sum(weights)
        pangkat = [-(w1/total_w), (w2/total_w), -(w3/total_w), -(w4/total_w), (w5/total_w)]
        
        data_kriteria = df_calc[['selling_price', 'year', 'km_driven', 'usia_motor', 'skor_kondisi']].values
        S = np.prod(np.power(data_kriteria, pangkat), axis=1)
        df_calc['Skor_V'] = S / np.sum(S)
        
        # Tabel Hasil Akhir yang di-Sort (Ketentuan 2e)
        st.subheader("2. Tabel Hasil Perangkingan")
        hasil_rank = df_calc[['name', 'year', 'selling_price', 'km_driven', 'Skor_V']].sort_values(by='Skor_V', ascending=False)
        st.dataframe(hasil_rank.reset_index(drop=True), use_container_width=True)
        
        st.success(f"Rekomendasi Utama: **{hasil_rank.iloc[0]['name']}**")

        # VISUALISASI ANALITIK (Sesuai Referensi Gambar)
        st.write("---")
        st.subheader("3. Visualisasi Analitik")
        
        sns.set_style("white")
        fig1, ax1 = plt.subplots(1, 3, figsize=(18, 6))

        # A. Top 10 Motor Terbaik (Bar Chart)
        sns.barplot(data=hasil_rank.head(10), x='Skor_V', y='name', ax=ax1[0], palette='viridis')
        ax1[0].set_title("Top 10 Motor Terbaik (WP)")

        # B. Korelasi Harga vs Skor (Scatter Plot)
        sns.scatterplot(data=df_calc, x='selling_price', y='Skor_V', ax=ax1[1], alpha=0.5, color='#7fb3d5')
        ax1[1].set_title("Korelasi Harga vs Skor Akhir")

        # C. Distribusi Tahun (Histogram)
        sns.histplot(df_calc['year'], bins=15, ax=ax1[2], kde=True, color='#fbc02d', edgecolor='black')
        ax1[2].set_title("Distribusi Tahun Kendaraan")
        st.pyplot(fig1)

        # Visualisasi Tren Tambahan
        st.write("---")
        st.subheader("4. Analisis Tren Distribusi")

        # D. Jumlah Motor Per Tahun
        df_year_count = df_calc.groupby('year').size().reset_index(name='jumlah')
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        plt.plot(df_year_count['year'], df_year_count['jumlah'], marker='o', color='#008080')
        plt.title("Jumlah Motor Berdasarkan Tahun Produksi")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig2)

        # E. Distribusi Harga
        price_dist = pd.cut(df_calc['selling_price'], bins=20).value_counts().sort_index()
        bin_mids = [b.mid for b in price_dist.index]
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        plt.plot(bin_mids, price_dist.values, marker='o', color='blue')
        plt.title("Distribusi Harga Motor")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig3)

        # F. Rata-Rata Harga per Tahun
        df_price_avg = df_calc.groupby('year')['selling_price'].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        plt.plot(df_price_avg['year'], df_price_avg['selling_price'], marker='s', color='green', linestyle='--')
        plt.title("Rata-Rata Harga Motor Berdasarkan Tahun")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig4)

# --- 6. HALAMAN 3: PROFIL KELOMPOK (Ketentuan 2a) ---
elif page == "Halaman Profil Kelompok":
    st.header("Profil Kelompok")
    st.write("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("### Anggota 1")
        st.write("Nama: **DWIRIZKI ADITHYA PUTRA**")
        st.write("NIM: **123240021**")
    with col_b:
        st.info("### Anggota 2")
        st.write("Nama: **MUAMMAR HAQQI WAHYATMA**")
        st.write("NIM: **123240204**")
    
    st.write("---")
    st.subheader("Informasi Proyek")
    st.write("- **Metode SPK:** Weighted Product (WP)")
    st.write("- **Dataset:** Motorcycle Dataset (nehalbirla/motorcycle-dataset)")
    st.write("- **Kriteria:** Harga Jual, Tahun Produksi, Jarak Tempuh, Usia Kendaraan, dan Estimasi Kondisi.")
  