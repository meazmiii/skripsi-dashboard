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
    # Nama file disesuaikan persis dengan folder di GitHub kamu
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
    # Data harus di-reshape ke (-1, 1) untuk Scaler
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    if len(scaled_data) < lookback:
        return None
        
    # Ambil urutan terakhir untuk input model
    last_sequence = scaled_data[-lookback:]
    
    # Reshape ke 3D (Batch, Timesteps, Feature) -> (1, lookback, 1)
    # Ini untuk memperbaiki error: expected shape=(None, 24, 1)
    last_sequence = last_sequence.reshape(1, lookback, 1)
    
    # Prediksi
    prediction_scaled = model.predict(last_sequence)
    
    # Kembalikan ke harga asli
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Sidebar Informasi
with st.sidebar:
    st.write("### Informasi Dashboard")
    st.info("""
    Dashboard ini menggunakan 3 model berbeda:
    - **Harian:** LSTM (Lookback 60)
    - **Mingguan:** TCN (Lookback 24)
    - **Bulanan:** TCN (Lookback 24)
    """)

# 5. Penarikan Data (Ambil 2 tahun agar data mingguan/bulanan mencukupi)
# Menggunakan period '2y' agar data historis cukup untuk lookback 24 bulan
df_raw = yf.download("BBCA.JK", period='2y')

if not df_raw.empty:
    # Mengambil data Close secara aman (Handle MultiIndex yfinance)
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_series = df_raw['Close'].iloc[:, 0]
    else:
        close_series = df_raw['Close']
    
    # Pastikan tidak ada data kosong agar grafik muncul sempurna
    close_series = close_series.dropna()

    # --- BAGIAN TABS UNTUK 3 TIMEFRAME ---
    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Prediksi Harga Harian")
        st.area_chart(close_series.tail(90)) # Menampilkan 90 hari terakhir
        
        last_p = float(close_series.iloc[-1])
        st.metric("Harga Terakhir (Harian)", f"Rp {last_p:,.2f}", f"Update: {df_raw.index[-1].date()}")
        
        if st.button('Mulai Prediksi Harian'):
            with st.spinner('Menganalisis pola harian...'):
                hasil = predict_stock(model_h, close_series.values, lookback=60)
                if hasil:
                    st.success(f"### Prediksi Harga Hari Besok: Rp {hasil:,.2f}")
                    st.balloons()
                else:
                    st.error("Data tidak cukup (Butuh minimal 60 hari).")

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Prediksi Harga Mingguan")
        # Resample data ke Mingguan
        df_weekly = close_series.resample('W-MON').last().dropna()
        st.area_chart(df_weekly.tail(52)) # Menampilkan 52 minggu terakhir
        
        last_p_w = float(df_weekly.iloc[-1])
        st.metric("Harga Penutupan Minggu Ini", f"Rp {last_p_w:,.2f}")
        
        if st.button('Mulai Prediksi Mingguan'):
            with st.spinner('Menganalisis pola mingguan...'):
                # Berdasarkan error sebelumnya, lookback diubah ke 24
                hasil = predict_stock(model_m, df_weekly.values, lookback=24)
                if hasil:
                    st.success(f"### Prediksi Harga Minggu Depan: Rp {hasil:,.2f}")
                else:
                    st.error("Data mingguan tidak cukup (Butuh minimal 24 minggu).")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Prediksi Harga Bulanan")
        # Resample data ke Bulanan
        df_monthly = close_series.resample('M').last().dropna()
        st.area_chart(df_monthly.tail(24)) # Menampilkan 24 bulan terakhir
        
        last_p_m = float(df_monthly.iloc[-1])
        st.metric("Harga Penutupan Bulan Ini", f"Rp {last_p_m:,.2f}")
        
        if st.button('Mulai Prediksi Bulanan'):
            with st.spinner('Menganalisis pola bulanan...'):
                # Lookback disesuaikan ke 24 sesuai arsitektur model bulanan kamu
                hasil = predict_stock(model_b, df_monthly.values, lookback=24)
                if hasil:
                    st.success(f"### Prediksi Harga Bulan Depan: Rp {hasil:,.2f}")
                else:
                    st.error("Data bulanan tidak cukup (Butuh minimal 24 bulan).")

    # Bagian Tabel di bawah semua Tab
    with st.expander("Lihat Detail Tabel Data Historis"):
        st.dataframe(df_raw.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Gagal mengambil data dari Yahoo Finance. Periksa koneksi internet.")
