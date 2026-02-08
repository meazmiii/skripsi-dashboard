import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Real-time Stock Prediction BBCA (LSTM & TCN)")

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
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
    last_sequence = last_sequence.reshape(1, lookback, 1) # Reshape ke 3D
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Sidebar Informasi
with st.sidebar:
    st.write("### Informasi Dashboard")
    st.info("Bandingkan harga aktual pasar saat ini dengan hasil prediksi model AI.")

# 5. Penarikan Data (5 tahun agar mencukupi semua timeframe)
df_raw = yf.download("BBCA.JK", period='5y')

if not df_raw.empty:
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_series = df_raw['Close'].iloc[:, 0]
    else:
        close_series = df_raw['Close']
    
    close_series = close_series.dropna()

    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan Harian")
        
        # Mengambil Harga Terakhir
        last_p = float(close_series.iloc[-1])
        last_d = df_raw.index[-1].date()

        # Layout Kolom untuk Aktual vs Prediksi
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Harga Aktual Terakhir", value=f"Rp {last_p:,.2f}", delta=f"Per Tanggal: {last_d}")
        
        if st.button('Jalankan Prediksi Harian'):
            with st.spinner('Menghitung prediksi...'):
                hasil = predict_stock(model_h, close_series.values, lookback=60)
                if hasil:
                    with col2:
                        st.metric(label="Hasil Prediksi Besok", value=f"Rp {hasil:,.2f}")
                    st.success(f"Analisis Selesai: Model memprediksi harga akan berada di kisaran Rp {hasil:,.2f}")
                    st.balloons()

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan Mingguan")
        df_weekly = close_series.resample('W-MON').last().dropna()
        last_p_w = float(df_weekly.iloc[-1])
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Harga Aktual Minggu Ini", value=f"Rp {last_p_w:,.2f}")
        
        if st.button('Jalankan Prediksi Mingguan'):
            with st.spinner('Menghitung prediksi...'):
                hasil = predict_stock(model_m, df_weekly.values, lookback=24)
                if hasil:
                    with c2:
                        st.metric(label="Hasil Prediksi Minggu Depan", value=f"Rp {hasil:,.2f}")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan Bulanan")
        df_monthly = close_series.resample('ME').last().dropna() # Pakai ME untuk standar 2026
        last_p_m = float(df_monthly.iloc[-1])
        
        k1, k2 = st.columns(2)
        with k1:
            st.metric(label="Harga Aktual Bulan Ini", value=f"Rp {last_p_m:,.2f}")
        
        if st.button('Jalankan Prediksi Bulanan'):
            with st.spinner('Menghitung prediksi...'):
                hasil = predict_stock(model_b, df_monthly.values, lookback=12)
                if hasil:
                    with k2:
                        st.metric(label="Hasil Prediksi Bulan Depan", value=f"Rp {hasil:,.2f}")

    with st.expander("Lihat Detail Tabel Data Historis"):
        st.dataframe(df_raw.sort_index(ascending=False), width='stretch')

else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
