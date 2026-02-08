import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    # Pastikan nama file ini sesuai dengan yang ada di repositori GitHub kamu
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi Universal (Fix Shape Error)
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    if len(scaled_data) < lookback:
        return None
        
    last_sequence = scaled_data[-lookback:]
    # Reshape ke 3D (1, lookback, 1) untuk model deep learning
    last_sequence = last_sequence.reshape(1, lookback, 1)
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Sidebar Informasi
with st.sidebar:
    st.write("### Informasi Dashboard")
    st.info("""
    Dashboard Prediksi BBCA:
    - **Harian:** Lookback 60
    - **Mingguan:** Lookback 24
    - **Bulanan:** Lookback 12
    """)

# 5. Penarikan Data (Ambil 5 tahun agar data mencukupi semua timeframe)
df_raw = yf.download("BBCA.JK", period='5y')

if not df_raw.empty:
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_series = df_raw['Close'].iloc[:, 0]
    else:
        close_series = df_raw['Close']
    
    close_series = close_series.dropna()

    # Menggunakan Tabs untuk memisahkan Timeframe
    tab1, tab2, tab3 = st.tabs(["📅 Harian", "🗓️ Mingguan", "📊 Bulanan"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan Harian (LSTM)")
        last_p = float(close_series.iloc[-1])
        last_d = df_raw.index[-1].date()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Harga Pasar Terakhir")
            st.metric(label=f"Aktual ({last_d})", value=f"Rp {last_p:,.2f}")
        
        if st.button('Jalankan Prediksi Harian'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_h, close_series.values, lookback=60)
                if hasil:
                    with col2:
                        st.markdown("### Hasil Prediksi Model")
                        st.metric(label="Estimasi Besok", value=f"Rp {hasil:,.2f}")
                    st.info(f"Analisis Selesai: Model memprediksi harga selanjutnya di Rp {hasil:,.2f}")

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan Mingguan (TCN)")
        # Resample ke akhir bulan (ME) sesuai standar terbaru 2026
        df_weekly = close_series.resample('W-MON').last().dropna()
        last_p_w = float(df_weekly.iloc[-1])
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Harga Aktual Minggu Ini")
            st.metric(label="Aktual (Mingguan)", value=f"Rp {last_p_w:,.2f}")
        
        if st.button('Jalankan Prediksi Mingguan'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_m, df_weekly.values, lookback=24)
                if hasil:
                    with c2:
                        st.markdown("### Hasil Prediksi Model")
                        st.metric(label="Estimasi Minggu Depan", value=f"Rp {hasil:,.2f}")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan Bulanan (TCN)")
        df_monthly = close_series.resample('ME').last().dropna()
        last_p_m = float(df_monthly.iloc[-1])
        
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("### Harga Aktual Bulan Ini")
            st.metric(label="Aktual (Bulanan)", value=f"Rp {last_p_m:,.2f}")
        
        if st.button('Jalankan Prediksi Bulanan'):
            with st.spinner('Menghitung...'):
                hasil = predict_stock(model_b, df_monthly.values, lookback=12)
                if hasil:
                    with k2:
                        st.markdown("### Hasil Prediksi Model")
                        st.metric(label="Estimasi Bulan Depan", value=f"Rp {hasil:,.2f}")

    st.markdown("---")
    with st.expander("Lihat Detail Tabel Data Historis"):
        st.dataframe(df_raw.sort_index(ascending=False), width='stretch')

else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
