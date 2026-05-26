import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
from datetime import datetime

# ---------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------
st.set_page_config(
    page_title="SPK Pemilihan Motor Bekas - WP",
    page_icon="🏍️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATASET                                                                                                              (3)  fungsi untuk memuat dataset dari Kaggle
# ---------------------------------------------------
@st.cache_data                     
def load_data():
    file_path = "BIKE DETAILS.csv"

    try:
        df_raw = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "nehalbirla/motorcycle-dataset",
            file_path
        )
        return df_raw.drop_duplicates()

    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return pd.DataFrame()

df = load_data()

# ---------------------------------------------------
# SIDEBAR NAVIGASI
# ---------------------------------------------------                                                                    (2a)  sidebar navigasi untuk memilih halaman
with st.sidebar:
    st.title("Menu Navigasi")

    page = st.radio(
        "Pilih Halaman:",
        [
            "Halaman Data",
            "Hitung SPK (WP)",
            "Halaman Profil Kelompok"
        ]
    )

    st.write("---")

# ===================================================
# HALAMAN DATASET                                                                                                        (2b)  halaman untuk menampilkan dataset
# ===================================================
if page == "Halaman Data":

    st.header("Dataset - Secondary Motorcycle")
    st.write(f"Total Dataset: **{len(df)} baris data**")                                                                  #(4) menampilkan jumlah total data yang dimuat
    st.dataframe(df, use_container_width=True)

# ===================================================
# HALAMAN HITUNG SPK
# ===================================================
elif page == "Hitung SPK (WP)":

    st.header("Sistem Pendukung Keputusan")
    st.subheader("Metode Weighted Product (WP)")                                                                        #(5) Judul Halaman

    if df.empty:
        st.warning("Dataset kosong atau gagal dimuat.")
        st.stop()

    # ---------------------------------------------------
    # PREPROCESSING DATA
    # ---------------------------------------------------
    df_calc = df.copy()

    tahun_sekarang = datetime.now().year

    df_calc["usia_motor"] = tahun_sekarang - df_calc["year"]

    df_calc["skor_kondisi"] = (
        100
        - (df_calc["km_driven"] / 1500)
        - (df_calc["usia_motor"] * 1.5)
    )

    df_calc["skor_kondisi"] = df_calc["skor_kondisi"].clip(lower=10)

    # ---------------------------------------------------
    # INPUT BOBOT                                                                                                       (2c)  input bobot kriteria
    # ---------------------------------------------------
    st.write("---")
    st.subheader("1. Input Bobot Kriteria")

    col1, col2 = st.columns(2)

    with col1:
        w1 = st.slider("Bobot Harga Jual (C1 - Cost)", 1, 5, 5)
        w2 = st.slider("Bobot Tahun Produksi (C2 - Benefit)", 1, 5, 4)
        w3 = st.slider("Bobot Jarak Tempuh/KM (C3 - Cost)", 1, 5, 3)

    with col2:
        w4 = st.number_input("Bobot Usia Kendaraan (C4 - Cost)", 1, 5, 2)
        w5 = st.number_input("Bobot Estimasi Kondisi (C5 - Benefit)", 1, 5, 4)

    st.info("""
    Keterangan Kriteria:
    - C1 Harga Jual = Cost
    - C2 Tahun Produksi = Benefit
    - C3 Jarak Tempuh/KM = Cost
    - C4 Usia Kendaraan = Cost
    - C5 Estimasi Kondisi = Benefit
    """)

    # ---------------------------------------------------
    # TOMBOL HITUNG                                                                                             (2d)  tombol untuk memulai perhitungan dan menampilkan hasil
    # ---------------------------------------------------
    if st.button("Hitung Peringkat Sekarang"):

        st.write("---")

        # ===================================================
        # PERHITUNGAN METODE WP
        # ===================================================
        weights = np.array([w1, w2, w3, w4, w5])
        total_w = sum(weights)

        pangkat = [
            -(w1 / total_w),
            (w2 / total_w),
            -(w3 / total_w),
            -(w4 / total_w),
            (w5 / total_w)
        ]

        data_kriteria = df_calc[                                                                                        #(4) mengambil data kriteria yang akan dihitung skor WP
            [
                "selling_price",
                "year",
                "km_driven",
                "usia_motor",
                "skor_kondisi"
            ]
        ].values

        S = np.prod(np.power(data_kriteria, pangkat), axis=1)                                                          #(5) menghitung skor WP untuk setiap alternatif

        df_calc["Skor_V"] = S / np.sum(S)

        # ===================================================
        # TABEL HASIL PERANGKINGAN                                                                                      (2e)  tabel untuk menampilkan hasil perangkingan berdasarkan skor WP
        # ===================================================
        st.subheader("2. Tabel Hasil Perangkingan")

        hasil_rank = df_calc[
            [
                "name",
                "year",
                "selling_price",
                "km_driven",
                "usia_motor",
                "skor_kondisi",
                "Skor_V"
            ]
        ].sort_values(
            by="Skor_V",
            ascending=False
        )

        st.dataframe(
            hasil_rank.reset_index(drop=True),
            use_container_width=True
        )

        st.success(
            f"Rekomendasi Utama: **{hasil_rank.iloc[0]['name']}**"
        )

        # ===================================================
        # VISUALISASI ANALITIK MODEL LINE CHART
        # ===================================================
        st.write("---")
        st.subheader("3. Visualisasi Analitik")

        # ==========================================
        # GRAFIK 1 - TOP 10 MOTOR TERBAIK
        # ==========================================
        st.markdown("### 1. Top 10 Motor Terbaik Berdasarkan Skor WP")

        top10 = hasil_rank.head(10).reset_index(drop=True)

        fig1, ax1 = plt.subplots(figsize=(14, 5))

        ax1.plot(
            top10["name"],
            top10["Skor_V"],
            marker="s",
            linestyle="--",
            color="green"
        )

        ax1.set_title("Top 10 Motor Terbaik Berdasarkan Skor WP")
        ax1.set_xlabel("Nama Motor")
        ax1.set_ylabel("Skor WP")
        ax1.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig1)

        # ==========================================
        # GRAFIK 2 - Harga Vs Rata-Rata Skor WP
        # ==========================================
        st.markdown("### 2. Harga Vs Rata-Rata Skor WP")

        df_harga = (
            df_calc.groupby("selling_price")["Skor_V"]
            .mean()
            .reset_index()
            .sort_values(by="selling_price")
        )

        fig2, ax2 = plt.subplots(figsize=(14, 5))

        ax2.plot(
            df_harga["selling_price"],
            df_harga["Skor_V"],
            marker="s",
            linestyle="--",
            color="green"
        )

        ax2.set_title("Harga Vs Rata-Rata Skor WP")
        ax2.set_xlabel("Harga Jual")
        ax2.set_ylabel("Rata-Rata Skor WP")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig2)

        # ==========================================
        # GRAFIK 3 - JUMLAH MOTOR BERDASARKAN TAHUN
        # ==========================================
        st.markdown("### 3. Jumlah Motor Berdasarkan Tahun Produksi")

        df_tahun = (
            df_calc.groupby("year")
            .size()
            .reset_index(name="jumlah")
            .sort_values(by="year")
        )

        fig3, ax3 = plt.subplots(figsize=(14, 5))

        ax3.plot(
            df_tahun["year"],
            df_tahun["jumlah"],
            marker="s",
            linestyle="--",
            color="green"
        )

        ax3.set_title("Jumlah Motor Berdasarkan Tahun Produksi")
        ax3.set_xlabel("Tahun")
        ax3.set_ylabel("Jumlah Motor")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig3)

        # ==========================================
        # GRAFIK 4 - DISTRIBUSI HARGA MOTOR
        # ==========================================
        st.markdown("### 4. Distribusi Harga Motor")

        price_dist = (
            pd.cut(
                df_calc["selling_price"],
                bins=20
            )
            .value_counts()
            .sort_index()
        )

        bin_mids = [b.mid for b in price_dist.index]

        fig4, ax4 = plt.subplots(figsize=(14, 5))

        ax4.plot(
            bin_mids,
            price_dist.values,
            marker="s",
            linestyle="--",
            color="green"
        )

        ax4.set_title("Distribusi Harga Motor")
        ax4.set_xlabel("Harga Motor")
        ax4.set_ylabel("Jumlah Data")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig4)

        # ==========================================
        # GRAFIK 5 - RATA-RATA HARGA BERDASARKAN TAHUN
        # ==========================================
        st.markdown("### 5. Rata-Rata Harga Motor Berdasarkan Tahun")

        df_price_avg = (
            df_calc.groupby("year")["selling_price"]
            .mean()
            .reset_index()
            .sort_values(by="year")
        )

        fig5, ax5 = plt.subplots(figsize=(14, 5))

        ax5.plot(
            df_price_avg["year"],
            df_price_avg["selling_price"],
            marker="s",
            linestyle="--",
            color="green"
        )

        ax5.set_title("Rata-Rata Harga Motor Berdasarkan Tahun")
        ax5.set_xlabel("Tahun")
        ax5.set_ylabel("Rata-Rata Harga")
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig5)

        # ==========================================
        # GRAFIK 6 - RATA-RATA KM BERDASARKAN TAHUN
        # ==========================================
        st.markdown("### 6. Rata-Rata Kilometer Berdasarkan Tahun")

        df_km_avg = (
            df_calc.groupby("year")["km_driven"]
            .mean()
            .reset_index()
            .sort_values(by="year")
        )

        fig6, ax6 = plt.subplots(figsize=(14, 5))

        ax6.plot(
            df_km_avg["year"],
            df_km_avg["km_driven"],
            marker="s",
            linestyle="--",
            color="green"
        )

        ax6.set_title("Rata-Rata Kilometer Berdasarkan Tahun")
        ax6.set_xlabel("Tahun")
        ax6.set_ylabel("Rata-Rata KM")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig6)

# ===================================================
# HALAMAN PROFIL KELOMPOK
# ===================================================
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
    st.write("- **Dataset:** Motorcycle Dataset dari Kaggle")
    st.write("- **Kriteria:** Harga Jual, Tahun Produksi, Jarak Tempuh, Usia Kendaraan, Estimasi Kondisi")