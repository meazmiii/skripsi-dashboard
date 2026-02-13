import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from streamlit_autorefresh import st_autorefresh

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi BBCA", layout="wide")

# Jam Real-time (Refresh setiap 1 detik)
st_autorefresh(interval=1000, key="clock_refresh")
tz_jkt = pytz.timezone('Asia/Jakarta')
now_jkt = datetime.now(tz_jkt)

st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

# --- PERBAIKAN: Kalimat Tetap Sama, Ukuran Diperbesar, Posisi Sejajar ---
st.write("") # Memberi ruang sedikit
col_jam, col_tgl = st.columns([1, 1])

with col_jam:
    # Menggunakan HTML sederhana di dalam markdown agar font-size bisa diatur besar
    st.markdown(f"## **Waktu Real-time:** <span style='font-size: 38px;'>`{now_jkt.strftime('%H:%M:%S')}`</span> **WIB**", unsafe_allow_html=True)

with col_tgl:
    # Menggunakan HTML sederhana untuk sejajar kanan dan font-size besar
    st.markdown(f"<div style='text-align: right;'><h2><b>Tanggal:</b> <span style='font-size: 38px;'><code>{now_jkt.strftime('%d-%m-%Y')}</code></span></h2></div>", unsafe_allow_html=True)

# --- PERBAIKAN: CSS Khusus Agar Tabel Tidak Gelap ---
st.markdown("""
    <style>
    /* Memaksa semua teks di dalam tabel/dataframe agar berwarna putih terang */
    .stDataFrame, [data-testid="stTable"] {
        color: #FFFFFF !important;
    }
    th {
        color: #FFFFFF !important;
        background-color: #31333F !important;
    }
    td {
        color: #FFFFFF !important;
    }
    /* Memperbaiki tampilan expander agar kontras */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: #262730 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Fungsi Load Model & Data
@st.cache_resource
def load_models():
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

@st.cache_data(ttl=3600)
def get_data():
    df = yf.download("BBCA.JK", period='5y')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    model_h, model_m, model_b = load_models()
    df_all = get_data()
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

# 3. Fungsi Prediksi
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    if len(scaled_data) < lookback: return None
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence)
    return scaler.inverse_transform(prediction_scaled)[0][0]

# --- CALLBACK UNTUK TOMBOL ---
def run_pred_h():
    st.session_state.res_h = predict_stock(model_h, df_all['Close'].dropna().values, 60)

def run_pred_w(data_w):
    st.session_state.res_w = predict_stock(model_m, data_w, 24)

def run_pred_m(data_m):
    st.session_state.res_m = predict_stock(model_b, data_m, 12)

# 4. Tampilan Utama
if not df_all.empty:
    close_series = df_all['Close'].dropna()
    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian (LSTM)")
        last_p = float(close_series.iloc[-1])
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, 60)

        c1, c2 = st.columns(2)
        with c1: st.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}")
        with c2: st.metric("Prediksi LSTM", f"Rp {pred_today:,.2f}")
        
        st.write("### 🕒 Historis Akurasi (5 Hari Bursa Terakhir)")
        history_h = []
        for i in range(1, 6):
            t_idx = -i
            act_val = close_series.iloc[t_idx]
            p_val = predict_stock(model_h, close_series.iloc[:t_idx].values, 60)
            history_h.append({
                "Tanggal": close_series.index[t_idx].date(),
                "Harga Aktual": f"Rp {act_val:,.2f}",
                "Prediksi LSTM": f"Rp {p_val:,.2f}",
                "Selisih (Rp)": f"{abs(act_val - p_val):,.2f}"
            })
        st.table(pd.DataFrame(history_h))

        st.button('Jalankan Prediksi LSTM (Besok)', on_click=run_pred_h, key='btn_h')
        if 'res_h' in st.session_state:
            st.success(f"### Estimasi Harga LSTM Besok: Rp {st.session_state.res_h:,.2f}")

        with st.expander("Lihat Data Historis Harian Lengkap"):
            st.dataframe(df_all.sort_index(ascending=False), use_container_width=True)

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan (TCN)")
        df_w = df_all.resample('W-MON').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_w = float(df_w['Close'].iloc[-1])
        pred_w = predict_stock(model_m, df_w['Close'].values[:-1], 24)

        col1, col2 = st.columns(2)
        with col1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
        with col2: st.metric("Prediksi TCN", f"Rp {pred_w:,.2f}")

        st.button('Jalankan Prediksi TCN (Minggu Depan)', on_click=run_pred_w, args=(df_w['Close'].values,), key='btn_w')
        if 'res_w' in st.session_state:
            st.success(f"### Estimasi Harga TCN Minggu Depan: Rp {st.session_state.res_w:,.2f}")

        with st.expander("Lihat Data Historis Mingguan Lengkap"):
            st.dataframe(df_w.sort_index(ascending=False), use_container_width=True)

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan (TCN)")
        df_m = df_all.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        last_p_m = float(df_m['Close'].iloc[-1])
        pred_m = predict_stock(model_b, df_m['Close'].values[:-1], 12)

        k1, k2 = st.columns(2)
        with k1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
        with k2: st.metric("Prediksi TCN", f"Rp {pred_m:,.2f}")

        st.button('Jalankan Prediksi TCN (Bulan Depan)', on_click=run_pred_m, args=(df_m['Close'].values,), key='btn_m')
        if 'res_m' in st.session_state:
            st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {st.session_state.res_m:,.2f}")

        with st.expander("Lihat Data Historis Bulanan Lengkap"):
            st.dataframe(df_m.sort_index(ascending=False), use_container_width=True)
# 1. Definisikan dulu variabelnya di luar try
ticker = 'BBCA.JK' 

try:
    # 2. Pastikan baris ini MENJOROK ke dalam (indentasi)
    df_ext = yf.download(ticker, start='2025-12-01', end='2026-02-02', progress=False)
    
    if df_ext.empty:
        st.error("Data tidak ditemukan atau kena Rate Limit. Coba lagi nanti.")
    else:
        st.success("Data berhasil dimuat!")
        # Lanjutkan proses plot atau prediksi di sini
        
except Exception as e:
    st.error(f"Terjadi error: {e}")

# --- Copyright Cerah & Terang ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #444;">
        <p style="margin:0; font-size: 14px; color: #DDDDDD;">© 2026 Skripsi Informatika - Universitas AMIKOM Yogyakarta</p>
        <p style="margin:5px 0; font-size: 18px; font-weight: bold; color: #FFFFFF;">AZMI AZIZ | 22.11.4903</p>
        <p style="margin:0; font-size: 15px;">
            <a href="https://www.instagram.com/_azmiazzz/?hl=id" target="_blank" style="color: #00AAFF; text-decoration: none; font-weight: bold;">Instagram: @_azmiazzz</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)





