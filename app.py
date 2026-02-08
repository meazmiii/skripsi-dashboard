import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman agar tampilan lebar
st.set_page_config(page_title="Stock Prediction BBCA", layout="wide")
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    # Nama file diselaraskan dengan folder GitHub kamu
    m_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    m_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    m_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return m_harian, m_mingguan, m_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi dengan Perbaikan Dimensi (Fix Shape Error)
def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    if len(scaled_data) < lookback:
        return None
        
    last_sequence = scaled_data[-lookback:]
    # Reshape ke 3D (1, lookback, 1) agar kompatibel dengan model
    last_sequence = last_sequence.reshape(1, lookback, 1)
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Penarikan Data Historis (5 tahun)
df_raw = yf.download("BBCA.JK", period='5y')

if not df_raw.empty:
    # Mengambil kolom Close secara aman
    if isinstance(df_raw.columns, pd.MultiIndex):
        close_series = df_raw['Close'].iloc[:, 0]
    else:
        close_series = df_raw['Close']
    
    close_series = close_series.dropna()

    # Navigasi Timeframe menggunakan Tab
    tab1, tab2, tab3 = st.tabs(["📅 Harian", "🗓️ Mingguan", "📊 Bulanan"])

# --- TAB 1: HARIAN ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian")
        
        # 1. Ambil data aktual terakhir dan data sebelumnya
        last_p = float(close_series.iloc[-1])
        last_d = df_raw.index[-1].date()
        
        # 2. Hitung 'Prediksi Hari Ini' menggunakan data kemarin (Backtesting Real-time)
        # Kita ambil data sampai H-1 untuk melihat apa kata model tentang harga HARI INI
        data_minus_1 = close_series.iloc[:-1].values
        prediksi_hari_ini = predict_stock(model_h, data_minus_1, lookback=60)

        # 3. Tampilkan Perbandingan (Kiri: Aktual, Kanan: Prediksi Model untuk Hari yang Sama)
        st.markdown("### Perbandingan Akurasi (Data Terkini)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"Harga Aktual ({last_d})", value=f"Rp {last_p:,.2f}")
        with c2:
            st.metric(label=f"Prediksi AI (untuk {last_d})", value=f"Rp {prediksi_hari_ini:,.2f}")
        
        st.info("💡 Bagian ini membandingkan harga asli bursa dengan hasil prediksi model menggunakan data hari sebelumnya.")

        st.markdown("---")
        
        # 4. Tombol Prediksi Masa Depan dipindah ke bawah
        st.markdown("### Prediksi Harga Besok")
        if st.button('Jalankan Prediksi Masa Depan'):
            with st.spinner('Menghitung harga besok...'):
                hasil_besok = predict_stock(model_h, close_series.values, lookback=60)
                if hasil_besok:
                    st.success(f"### Estimasi Harga Hari Bursa Selanjutnya: Rp {hasil_besok:,.2f}")

    # --- TAB 2: MINGGUAN ---
    with tab2:
        st.subheader("Analisis Perbandingan Mingguan (TCN)")
        df_weekly = close_series.resample('ME').last().dropna()
        last_p_w = float(df_weekly.iloc[-1])
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Harga Penutupan Minggu Ini")
            st.metric(label="Aktual (Mingguan)", value=f"Rp {last_p_w:,.2f}")
        
        with c2:
            st.markdown("### Prediksi Minggu Depan")
            if st.button('Hitik Prediksi Mingguan'):
                hasil = predict_stock(model_m, df_weekly.values, lookback=24)
                if hasil:
                    st.metric(label="Estimasi Minggu Depan", value=f"Rp {hasil:,.2f}")
                    st.success(f"Model memprediksi harga akan bergerak ke Rp {hasil:,.2f}")

    # --- TAB 3: BULANAN ---
    with tab3:
        st.subheader("Analisis Perbandingan Bulanan (TCN)")
        df_monthly = close_series.resample('ME').last().dropna()
        last_p_m = float(df_monthly.iloc[-1])
        
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("### Harga Penutupan Bulan Ini")
            st.metric(label="Aktual (Bulanan)", value=f"Rp {last_p_m:,.2f}")
        
        with k2:
            st.markdown("### Prediksi Bulan Depan")
            if st.button('Hitung Prediksi Bulanan'):
                hasil = predict_stock(model_b, df_monthly.values, lookback=12)
                if hasil:
                    st.metric(label="Estimasi Bulan Depan", value=f"Rp {hasil:,.2f}")
                    st.success(f"Model memprediksi harga akan bergerak ke Rp {hasil:,.2f}")

    st.markdown("---")
    with st.expander("Lihat Data Historis Lengkap"):
        st.dataframe(df_raw.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Gagal menyambung ke data Yahoo Finance.")



