import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi BBCA", layout="wide")
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

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

# 3. Fungsi Prediksi Universal
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    if len(scaled_data) < lookback:
        return None
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence)
    return scaler.inverse_transform(prediction_scaled)[0][0]

# 4. Penarikan Data (5 tahun)
df_raw = yf.download("BBCA.JK", period='5y')

if not df_raw.empty:
    close_series = df_raw['Close'].iloc[:, 0] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw['Close']
    close_series = close_series.dropna()

    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian")
        last_p = float(close_series.iloc[-1])
        last_d = df_raw.index[-1].date()
        
        # Hitung Prediksi untuk HARI INI (menggunakan data s/d kemarin)
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, lookback=60)

        st.markdown("### Perbandingan Akurasi (Data Terkini)")
        c1, c2 = st.columns(2)
        with c1: st.metric(f"Harga Aktual ({last_d})", f"Rp {last_p:,.2f}")
        with c2: st.metric(f"Prediksi AI (untuk {last_d})", f"Rp {pred_today:,.2f}")
        
        st.info("💡 Membandingkan harga asli bursa dengan tebakan model menggunakan data hari sebelumnya.")
        st.markdown("---")
        if st.button('Jalankan Prediksi Masa Depan (Harian)'):
            hasil = predict_stock(model_h, close_series.values, lookback=60)
            st.success(f"### Estimasi Harga Besok: Rp {hasil:,.2f}")

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan")
        df_w = close_series.resample('W-MON').last().dropna()
        last_p_w = float(df_w.iloc[-1])
        
        # Prediksi minggu ini (pakai data s/d minggu lalu)
        pred_w = predict_stock(model_m, df_w.iloc[:-1].values, lookback=24)

        col1, col2 = st.columns(2)
        with col1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
        with col2: st.metric("Prediksi AI (Minggu Ini)", f"Rp {pred_w:,.2f}")
        
        st.markdown("---")
        if st.button('Jalankan Prediksi Minggu Depan'):
            hasil = predict_stock(model_m, df_w.values, lookback=24)
            st.success(f"### Estimasi Harga Minggu Depan: Rp {hasil:,.2f}")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan")
        df_m = close_series.resample('ME').last().dropna()
        last_p_m = float(df_m.iloc[-1])
        
        # Prediksi bulan ini (pakai data s/d bulan lalu)
        pred_m = predict_stock(model_b, df_m.iloc[:-1].values, lookback=12)

        k1, k2 = st.columns(2)
        with k1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
        with k2: st.metric("Prediksi AI (Bulan Ini)", f"Rp {pred_m:,.2f}")
        
        st.markdown("---")
        if st.button('Jalankan Prediksi Bulan Depan'):
            hasil = predict_stock(model_b, df_m.values, lookback=12)
            st.success(f"### Estimasi Harga Bulan Depan: Rp {hasil:,.2f}")

    with st.expander("Lihat Data Historis Lengkap"):
        st.dataframe(df_raw.sort_index(ascending=False), use_container_width=True)
else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
