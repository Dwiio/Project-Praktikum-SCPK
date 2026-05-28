import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from mysql.connector import Error

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="SPK Pemilihan Motor Bekas - WP",
    page_icon="🏍️",
    layout="wide"
)

# =========================
# KONEKSI DATABASE MYSQL
# =========================
def get_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # isi jika MySQL kamu memakai password
            database="spk_motor"
        )
    except Error as e:
        st.error(f"Gagal koneksi ke MySQL: {e}")
        return None


def load_data_dari_mysql():
    conn = get_connection()
    if conn:
        query = """
            SELECT 
                id,
                name,
                selling_price,
                year,
                seller_type,
                owner,
                km_driven,
                ex_showroom_price,
                usia_motor,
                skor_kondisi
            FROM motorcycles
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    return pd.DataFrame()


def tambah_motor(name, selling_price, year, seller_type, owner, km_driven, ex_showroom_price=""):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        usia_motor = 2026 - int(year)
        skor_kondisi = max(10, 100 - (int(km_driven) / 1500) - (usia_motor * 1.5))

        cursor.execute("""
            INSERT INTO motorcycles
            (name, selling_price, year, seller_type, owner, km_driven, ex_showroom_price, usia_motor, skor_kondisi)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            name,
            int(selling_price),
            int(year),
            seller_type,
            owner,
            int(km_driven),
            str(ex_showroom_price),
            int(usia_motor),
            float(skor_kondisi)
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True
    return False


def update_motor(motor_id, name, selling_price, year, km_driven, seller_type, owner):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        usia_motor = 2026 - int(year)
        skor_kondisi = max(10, 100 - (int(km_driven) / 1500) - (usia_motor * 1.5))

        cursor.execute("""
            UPDATE motorcycles
            SET name=%s,
                selling_price=%s,
                year=%s,
                seller_type=%s,
                owner=%s,
                km_driven=%s,
                usia_motor=%s,
                skor_kondisi=%s
            WHERE id=%s
        """, (
            name,
            int(selling_price),
            int(year),
            seller_type,
            owner,
            int(km_driven),
            int(usia_motor),
            float(skor_kondisi),
            int(motor_id)
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True
    return False


def hapus_motor(motor_id):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM motorcycles WHERE id=%s", (int(motor_id),))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    return False


# =========================
# PERHITUNGAN WP
# =========================
def hitung_weighted_product(df, bobot):
    df_hasil = df.copy()

    kriteria = [
        "selling_price",      # C1 cost
        "year",               # C2 benefit
        "km_driven",          # C3 cost
        "usia_motor",         # C4 cost
        "skor_kondisi"        # C5 benefit
    ]

    # Hindari nilai 0 karena WP memakai perkalian berpangkat
    df_hasil[kriteria] = df_hasil[kriteria].replace(0, 1)
    df_hasil[kriteria] = df_hasil[kriteria].astype(float)

    bobot = np.array(bobot, dtype=float)
    bobot_normal = bobot / bobot.sum()

    # Cost bernilai negatif, benefit bernilai positif
    pangkat = np.array([
        -bobot_normal[0],
        bobot_normal[1],
        -bobot_normal[2],
        -bobot_normal[3],
        bobot_normal[4]
    ])

    matriks = df_hasil[kriteria].values
    nilai_s = np.prod(np.power(matriks, pangkat), axis=1)
    nilai_v = nilai_s / nilai_s.sum()

    df_hasil["Nilai_S"] = nilai_s
    df_hasil["Skor_V"] = nilai_v
    df_hasil["Peringkat"] = df_hasil["Skor_V"].rank(ascending=False, method="first").astype(int)

    df_hasil = df_hasil.sort_values("Skor_V", ascending=False).reset_index(drop=True)
    df_hasil["Peringkat"] = range(1, len(df_hasil) + 1)

    return df_hasil


# =========================
# SIDEBAR NAVIGASI
# =========================
with st.sidebar:
    st.title("🏍️ Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        [
            "Halaman Data",
            "Kelola Database",
            "Hitung SPK (WP)",
            "Profil Kelompok"
        ]
    )

    st.write("---")
    st.caption("Database: MySQL / spk_motor")
    st.caption("Metode: Weighted Product")


# =========================
# HALAMAN DATA
# =========================
if page == "Halaman Data":
    st.header("📊 Halaman Data Motor Bekas")
    st.write("Dataset bersumber dari Kaggle dan disimpan ke database MySQL.")

    df = load_data_dari_mysql()

    if df.empty:
        st.warning("Database masih kosong. Import dataset terlebih dahulu melalui MySQL/phpMyAdmin.")
    else:
        st.success(f"Dataset memiliki {len(df)} baris data dan {len(df.columns)} kolom.")

        st.subheader("Dataset Mentah")
        st.dataframe(df, use_container_width=True)

        st.subheader("Ringkasan Dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Jumlah Data", len(df))
        col2.metric("Jumlah Kriteria", 5)
        col3.metric("Tahun Termuda", int(df["year"].max()))
        col4.metric("Tahun Tertua", int(df["year"].min()))

        st.info(
            "Kriteria yang digunakan: Harga Jual, Tahun Produksi, KM Driven, Usia Motor, dan Skor Kondisi."
        )


# =========================
# KELOLA DATABASE
# =========================
elif page == "Kelola Database":
    st.header("🛠️ Kelola Database Motor")
    st.write("Fitur CRUD untuk menambah, mengubah, dan menghapus data alternatif motor.")

    # Menampilkan notifikasi setelah proses tambah/edit/hapus berhasil
    if "notif_crud" in st.session_state:
        st.success(st.session_state["notif_crud"])
        del st.session_state["notif_crud"]

    tab1, tab2, tab3 = st.tabs(["➕ Tambah Data", "✏️ Edit Data", "🗑️ Hapus Data"])

    with tab1:
        st.subheader("Tambah Data Motor")
        with st.form("form_tambah"):
            nama = st.text_input("Nama Motor")

            col1, col2 = st.columns(2)
            with col1:
                harga = st.number_input("Harga Jual", min_value=1, value=10000000)
                tahun = st.number_input("Tahun Produksi", min_value=1990, max_value=2026, value=2020)
                seller = st.selectbox("Seller Type", ["Individual", "Dealer"])

            with col2:
                km = st.number_input("KM Driven", min_value=1, value=10000)
                owner = st.selectbox("Owner", ["1st owner", "2nd owner", "3rd owner", "4th owner or more"])
                showroom = st.text_input("Ex Showroom Price", value="")

            submit = st.form_submit_button("Simpan Data")

            if submit:
                if nama.strip() == "":
                    st.warning("Nama motor tidak boleh kosong.")
                else:
                    if tambah_motor(nama, harga, tahun, seller, owner, km, showroom):
                        st.session_state["notif_crud"] = f"✅ Data motor '{nama}' berhasil ditambahkan ke database."
                        st.rerun()

    with tab2:
        st.subheader("Edit Data Motor")
        df_edit = load_data_dari_mysql()

        if df_edit.empty:
            st.info("Belum ada data untuk diedit.")
        else:
            motor_id = st.selectbox("Pilih ID Motor", df_edit["id"].tolist())
            motor = df_edit[df_edit["id"] == motor_id].iloc[0]

            with st.form("form_edit"):
                nama_edit = st.text_input("Nama Motor", value=motor["name"])

                col1, col2 = st.columns(2)
                with col1:
                    harga_edit = st.number_input("Harga Jual", min_value=1, value=int(motor["selling_price"]))
                    tahun_edit = st.number_input("Tahun Produksi", min_value=1990, max_value=2026, value=int(motor["year"]))
                    seller_edit = st.selectbox(
                        "Seller Type",
                        ["Individual", "Dealer"],
                        index=0 if motor["seller_type"] == "Individual" else 1
                    )

                with col2:
                    km_edit = st.number_input("KM Driven", min_value=1, value=int(motor["km_driven"]))
                    owner_options = ["1st owner", "2nd owner", "3rd owner", "4th owner or more"]
                    owner_value = motor["owner"] if motor["owner"] in owner_options else "1st owner"
                    owner_edit = st.selectbox("Owner", owner_options, index=owner_options.index(owner_value))

                submit_edit = st.form_submit_button("Update Data")

                if submit_edit:
                    if update_motor(motor_id, nama_edit, harga_edit, tahun_edit, km_edit, seller_edit, owner_edit):
                        st.session_state["notif_crud"] = f"✅ Data motor ID {motor_id} berhasil diedit / diupdate."
                        st.rerun()

    with tab3:
        st.subheader("Hapus Data Motor")
        df_hapus = load_data_dari_mysql()

        if df_hapus.empty:
            st.info("Belum ada data untuk dihapus.")
        else:
            motor_id_hapus = st.selectbox("Pilih ID Motor", df_hapus["id"].tolist(), key="hapus")
            motor = df_hapus[df_hapus["id"] == motor_id_hapus].iloc[0]

            st.warning(f"Data yang akan dihapus: {motor['name']} - Tahun {motor['year']}")

            if st.button("Hapus Data"):
                if hapus_motor(motor_id_hapus):
                    st.session_state["notif_crud"] = f"✅ Data motor '{motor['name']}' berhasil dihapus dari database."
                    st.rerun()


# =========================
# HITUNG SPK WP
# =========================
elif page == "Hitung SPK (WP)":
    st.header("🧮 Perhitungan SPK Metode Weighted Product")

    df = load_data_dari_mysql()

    if df.empty:
        st.warning("Database kosong. Masukkan dataset ke database terlebih dahulu.")
    else:
        st.subheader("1. Kriteria Penilaian")

        kriteria_info = pd.DataFrame({
            "Kode": ["C1", "C2", "C3", "C4", "C5"],
            "Kriteria": ["Harga Jual", "Tahun Produksi", "KM Driven", "Usia Motor", "Skor Kondisi"],
            "Atribut": ["Cost", "Benefit", "Cost", "Cost", "Benefit"],
            "Keterangan": [
                "Semakin murah semakin baik",
                "Semakin baru semakin baik",
                "Semakin kecil KM semakin baik",
                "Semakin muda usia motor semakin baik",
                "Semakin tinggi skor kondisi semakin baik"
            ]
        })
        st.dataframe(kriteria_info, use_container_width=True)

        st.subheader("2. Input Bobot Dinamis")
        st.write("Ubah bobot sesuai prioritas. Nilai bobot akan dinormalisasi otomatis.")

        col1, col2 = st.columns(2)
        with col1:
            w1 = st.slider("Bobot C1 - Harga Jual", 1, 5, 5)
            w2 = st.slider("Bobot C2 - Tahun Produksi", 1, 5, 4)
            w3 = st.slider("Bobot C3 - KM Driven", 1, 5, 3)

        with col2:
            w4 = st.number_input("Bobot C4 - Usia Motor", min_value=1, max_value=5, value=2)
            w5 = st.number_input("Bobot C5 - Skor Kondisi", min_value=1, max_value=5, value=4)

        jumlah_tampil = st.selectbox("Jumlah Top Ranking yang Ditampilkan", [10, 20, 50, 100, "Semua"])

        if st.button("🔢 Hitung Ranking WP"):
            hasil = hitung_weighted_product(df, [w1, w2, w3, w4, w5])

            st.subheader("3. Tabel Hasil Perangkingan")

            kolom_hasil = [
                "Peringkat",
                "name",
                "selling_price",
                "year",
                "km_driven",
                "usia_motor",
                "skor_kondisi",
                "Nilai_S",
                "Skor_V"
            ]

            if jumlah_tampil == "Semua":
                tampil = hasil[kolom_hasil]
            else:
                tampil = hasil[kolom_hasil].head(int(jumlah_tampil))

            st.dataframe(tampil, use_container_width=True)

            terbaik = hasil.iloc[0]
            st.success(
                f"🏆 Rekomendasi Peringkat 1: {terbaik['name']} | "
                f"Skor V: {terbaik['Skor_V']:.6f}"
            )

            st.subheader("4. Visualisasi Data Analitik")
            

            # Menyiapkan data agregat
            top10 = hasil.head(10).copy()
            top10["label_motor"] = top10["Peringkat"].astype(str) + ". " + top10["name"].str[:25]

            harga_tahun = hasil.groupby("year", as_index=False)["selling_price"].mean()
            jumlah_tahun = hasil.groupby("year", as_index=False).size().rename(columns={"size": "jumlah_motor"})
            km_tahun = hasil.groupby("year", as_index=False)["km_driven"].mean()
            skor_tahun = hasil.groupby("year", as_index=False)["Skor_V"].mean()
            kondisi_tahun = hasil.groupby("year", as_index=False)["skor_kondisi"].mean()

            # Grafik 1: Top 10 skor WP
            st.write("**Grafik 1 - Top 10 Motor Berdasarkan Skor WP**")
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            ax1.plot(top10["label_motor"], top10["Skor_V"], marker="o")
            ax1.set_title("Top 10 Motor Terbaik Berdasarkan Skor WP", fontsize=18)
            ax1.set_xlabel("Alternatif Motor", fontsize=12)
            ax1.set_ylabel("Skor V", fontsize=12)
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

            # Grafik 2: Rata-rata harga motor per tahun
            st.write("**Grafik 2 - Rata-rata Harga Motor per Tahun**")
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            ax2.plot(harga_tahun["year"], harga_tahun["selling_price"], marker="o")
            ax2.set_title("Rata-rata Harga Motor Berdasarkan Tahun", fontsize=18)
            ax2.set_xlabel("Tahun Produksi", fontsize=12)
            ax2.set_ylabel("Rata-rata Harga", fontsize=12)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

            # Grafik 3: Jumlah motor per tahun
            st.write("**Grafik 3 - Jumlah Motor Berdasarkan Tahun Produksi**")
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            ax3.plot(jumlah_tahun["year"], jumlah_tahun["jumlah_motor"], marker="o")
            ax3.set_title("Jumlah Motor Berdasarkan Tahun Produksi", fontsize=18)
            ax3.set_xlabel("Tahun Produksi", fontsize=12)
            ax3.set_ylabel("Jumlah Motor", fontsize=12)
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

            # Grafik 4: Rata-rata KM driven per tahun
            st.write("**Grafik 4 - Rata-rata KM Driven per Tahun**")
            fig4, ax4 = plt.subplots(figsize=(14, 6))
            ax4.plot(km_tahun["year"], km_tahun["km_driven"], marker="o")
            ax4.set_title("Rata-rata KM Driven Motor Berdasarkan Tahun", fontsize=18)
            ax4.set_xlabel("Tahun Produksi", fontsize=12)
            ax4.set_ylabel("Rata-rata KM Driven", fontsize=12)
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)

            # Grafik 5: Rata-rata skor kondisi dan skor WP per tahun
            st.write("**Grafik 5 - Rata-rata Skor Kondisi per Tahun**")
            fig5, ax5 = plt.subplots(figsize=(14, 6))
            ax5.plot(kondisi_tahun["year"], kondisi_tahun["skor_kondisi"], marker="o")
            ax5.set_title("Rata-rata Skor Kondisi Motor Berdasarkan Tahun", fontsize=18)
            ax5.set_xlabel("Tahun Produksi", fontsize=12)
            ax5.set_ylabel("Rata-rata Skor Kondisi", fontsize=12)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)



# =========================
# PROFIL KELOMPOK
# =========================
elif page == "Profil Kelompok":
    st.header("👥 Profil Kelompok")

    col1, col2 = st.columns(2)

    with col1:
        st.info("Anggota 1")
        st.write("Nama: **DWIRIZKI ADITHYA PUTRA**")
        st.write("NIM: **123240021**")

    with col2:
        st.info("Anggota 2")
        st.write("Nama: **MUAMMAR HAQQI WAHYATMA**")
        st.write("NIM: **123240204**")
