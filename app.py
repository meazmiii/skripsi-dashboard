import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIG HALAMAN ---
st.set_page_config(
    page_title="Dashboard Skripsi - Azmi Aziz",
    page_icon="📈",
    layout="wide"
)

# --- 2. SIDEBAR (PROFIL) ---
with st.sidebar:
    st.image("https://home.amikom.ac.id/media/2020/08/logo-amikom-t.png", width=250)
    st.title("Parameter Penelitian")
    st.info("""
    **Peneliti:** Azmi Aziz
    **Kampus:** AMIKOM Yogyakarta
    **Topik:** Komparasi LSTM vs TCN
    **Objek:** Saham BBCA (2010-2025)
    """)
    
    st.markdown("---")
    st.write("Dibuat dengan Python & Streamlit")

# --- 3. JUDUL UTAMA ---
st.title("📈 Analisis Komparasi Model Deep Learning")
st.markdown("""
Dashboard ini menampilkan hasil eksperimen perbandingan kinerja algoritma **Long Short-Term Memory (LSTM)** melawan **Temporal Convolutional Network (TCN)** dalam memprediksi harga saham **BBCA**.
""")
st.markdown("---")

# --- 4. LOAD DATA ---
try:
    df = pd.read_csv('Hasil_Eksperimen_Tuning.csv')
    
    # Paksa Urutan Timeframe (Harian -> Mingguan -> Bulanan)
    urutan_custom = ['Harian', 'Mingguan', 'Bulanan']
    df['Time Frame'] = pd.Categorical(df['Time Frame'], categories=urutan_custom, ordered=True)
    df = df.sort_values(by=['Time Frame', 'RMSE'])

    # --- 5. TOP LEVEL METRICS (KARTU SKOR) ---
    # Cari model dengan RMSE terendah (Juara 1 Dunia)
    best_overall = df.loc[df['RMSE'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏆 Model Terbaik (Juara Umum)",
            value=best_overall['Model'],
            delta=f"Timeframe {best_overall['Time Frame']}"
        )
    
    with col2:
        st.metric(
            label="📉 RMSE Terendah",
            value=f"{best_overall['RMSE']:.2f}",
            delta="Makin kecil makin bagus",
            delta_color="inverse"
        )
        
    with col3:
        st.metric(
            label="⚙️ Konfigurasi Terbaik",
            value=f"{best_overall['Units']} Units",
            delta=f"LR: {best_overall['LR']}"
        )

    # --- 6. GRAFIK UTAMA (INTERAKTIF PLOTLY) ---
    st.subheader("📊 Visualisasi Perbandingan Error (RMSE)")
    
    # Filter Data (Ambil yang terbaik per kategori)
    best_per_model = df.groupby(['Time Frame', 'Model'], observed=True).apply(lambda x: x.nsmallest(1, 'RMSE')).reset_index(drop=True)
    
    # Bikin Chart Keren
    fig = px.bar(
        best_per_model, 
        x="Time Frame", 
        y="RMSE", 
        color="Model",
        barmode="group",
        text_auto='.2f', # Menampilkan angka di atas batang
        color_discrete_map={"LSTM": "#3366CC", "TCN": "#FF9900"}, # Warna Biru vs Oranye Custom
        title="Komparasi RMSE: LSTM vs TCN (Lebih Rendah Lebih Baik)",
        height=500
    )
    
    # Update layout biar rapi
    fig.update_layout(
        xaxis_title="Periode Waktu",
        yaxis_title="Nilai RMSE (Rupiah)",
        legend_title="Jenis Algoritma",
        font=dict(size=14)
    )
    
    # Tampilkan Chart
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. DATA TABLE & INSIGHTS ---
    col_kiri, col_kanan = st.columns([1.5, 1])

    with col_kiri:
        st.subheader("📋 Tabel Data Detail")
        st.dataframe(
            df[['Time Frame', 'Model', 'RMSE', 'MAE', 'MAPE (%)', 'Units', 'LR']],
            use_container_width=True,
            hide_index=True
        )

    with col_kanan:
        st.subheader("💡 Insight Penting")
        st.warning(f"""
        **Temuan Utama:**
        1. **{best_overall['Model']}** terbukti lebih unggul dibandingkan kompetitornya.
        2. Timeframe **Harian** memberikan error paling kecil karena jumlah data latih lebih banyak.
        3. Model TCN cenderung kesulitan pada data Bulanan (RMSE Melonjak).
        """)

except FileNotFoundError:
    st.error("⚠️ File 'Hasil_Eksperimen_Tuning.csv' belum diupload!")
    st.info("Silakan upload file CSV hasil dari Google Colab ke folder yang sama dengan app.py")
