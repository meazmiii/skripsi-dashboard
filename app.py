import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi BBCA", layout="wide")
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

# --- FITUR JAM REAL-TIME ---
# Membuat placeholder untuk jam di bagian paling atas atau sidebar
ph_jam = st.empty()

# 2. Fungsi Load Model
@st.cache_resource
def load_models():
    # Sesuaikan dengan nama file model di folder kamu
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

# Logika Jam Real-time di Sidebar atau Pojok Atas
now = datetime.now()
ph_jam.markdown(f"**Waktu Sistem (Real-time):** `{now.strftime('%H:%M:%S')}` | **Tanggal:** `{now.strftime('%d-%m-%Y')}`")

if not df_raw.empty:
    close_series = df_raw['Close'].iloc[:, 0] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw['Close']
    close_series = close_series.dropna()

    tab1, tab2, tab3 = st.tabs(["📅 Harian (LSTM)", "🗓️ Mingguan (TCN)", "📊 Bulanan (TCN)"])

    # --- TAB 1: HARIAN (LSTM) ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian (LSTM)")
        
        last_p = float(close_series.iloc[-1])
        last_d = close_series.index[-1].date()
        pred_today = predict_stock(model_h, close_series.iloc[:-1].values, lookback=60)

        c1, c2 = st.columns(2)
        # Menambahkan keterangan waktu pada metrik harga aktual
        with c1: 
            st.metric(f"Harga Aktual Terakhir ({last_d})", f"Rp {last_p:,.2f}")
            st.caption(f"Diperbarui pada: {now.strftime('%H:%M:%S')} WIB")
        
        with c2: 
            st.metric(f"Prediksi LSTM (untuk {last_d})", f"Rp {pred_today:,.2f}")
        
        st.markdown("---")
        st.write("### 🕒 Historis Akurasi (5 Hari Bursa Terakhir)")
        history_list = []
        for i in range(1, 6):
            target_idx = -i
            actual_val = close_series.iloc[target_idx]
            actual_date = close_series.index[target_idx].date()
            input_data = close_series.iloc[:target_idx].values
            pred_val = predict_stock(model_h, input_data, lookback=60)
            history_list.append({
                "Tanggal": actual_date,
                "Harga Aktual": f"Rp {actual_val:,.2f}",
                "Prediksi LSTM": f"Rp {pred_val:,.2f}",
                "Selisih (Rp)": f"{abs(actual_val - pred_val):,.2f}"
            })
        st.table(pd.DataFrame(history_list))

        if st.button('Jalankan Prediksi LSTM (Besok)'):
            hasil = predict_stock(model_h, close_series.values, lookback=60)
            st.success(f"### Estimasi Harga LSTM Besok: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Harian"):
            st.dataframe(df_raw.sort_index(ascending=False))

    # --- TAB 2: MINGGUAN (TCN) ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan (TCN)")
        df_w = close_series.resample('W-MON').last().dropna()
        last_p_w = float(df_w.iloc[-1])
        pred_w = predict_stock(model_m, df_w.iloc[:-1].values, lookback=24)

        col1, col2 = st.columns(2)
        with col1: 
            st.metric(f"Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
            st.caption(f"Status Data: {now.strftime('%H:%M:%S')} WIB")
        with col2: st.metric("Prediksi TCN (Minggu Ini)", f"Rp {pred_w:,.2f}")
        
        st.markdown("---")
        if st.button('Jalankan Prediksi TCN (Minggu Depan)'):
            hasil = predict_stock(model_m, df_w.values, lookback=24)
            st.success(f"### Estimasi Harga TCN Minggu Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Mingguan"):
            st.dataframe(df_w.sort_index(ascending=False))

    # --- TAB 3: BULANAN (TCN) ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan (TCN)")
        df_m = close_series.resample('ME').last().dropna()
        last_p_m = float(df_m.iloc[-1])
        pred_m = predict_stock(model_b, df_m.iloc[:-1].values, lookback=12)

        k1, k2 = st.columns(2)
        with k1: 
            st.metric(f"Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
            st.caption(f"Status Data: {now.strftime('%H:%M:%S')} WIB")
        with k2: st.metric("Prediksi TCN (Bulan Ini)", f"Rp {pred_m:,.2f}")
        
        st.markdown("---")
        if st.button('Jalankan Prediksi TCN (Bulan Depan)'):
            hasil = predict_stock(model_b, df_m.values, lookback=12)
            st.success(f"### Estimasi Harga TCN Bulan Depan: Rp {hasil:,.2f}")

        with st.expander("Lihat Data Historis Bulanan"):
            st.dataframe(df_m.sort_index(ascending=False))

else:
    st.warning("Gagal mengambil data dari Yahoo Finance.")
