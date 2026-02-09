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

# --- PERBAIKAN: Waktu & Tanggal Diperbesar ---
st.markdown(
    f"""
    <div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border-left: 10px solid #00FF00; margin-bottom:20px;">
        <h1 style="color:white; margin:0; font-size: 50px; font-family: monospace;">{now_jkt.strftime('%H:%M:%S')} <span style="font-size: 20px;">WIB</span></h1>
        <h3 style="color:#AAAAAA; margin:0;">{now_jkt.strftime('%A, %d %B %Y')}</h3>
    </div>
    """, 
    unsafe_allow_html=True
)

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

        st.button('Jalankan Prediksi LSTM (Besok)', on_click=run_pred_h)
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

        st.button('Jalankan Prediksi TCN (Minggu Depan)', on_click=run_pred_w, args=(df_w['Close'].values,))
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

        st.button('Jalankan Prediksi TCN (Bulan Depan)', on_click=run_pred_m, args=(df_m['Close'].values,))
        if 'res_m' in st.session_state:
            st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {st.session_state.res_m:,.2f}")

        with st.expander("Lihat Data Historis Bulanan Lengkap"):
            st.dataframe(df_m.sort_index(ascending=False), use_container_width=True)

# --- ADDED: Footer Copyright & Identitas ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #777777; padding: 10px;">
        <p style="margin:0;">© 2026 Skripsi Informatika - Universitas AMIKOM Yogyakarta</p>
        <p style="margin:0; font-weight: bold; color: #AAAAAA;">AZMI AZIZ | 22.11.4903</p>
        <p style="margin:0;">Instagram: <a href="https://instagram.com/_azmiazzz" style="color: #00AAFF; text-decoration: none;">@_azmiazzz</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
