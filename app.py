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
    # Nama file disesuaikan persis dengan yang ada di folder GitHub kamu
    model_harian = load_model('Tuned_LSTM_Harian_U64_LR0.001_KN.h5', compile=False)
    model_mingguan = load_model('Tuned_TCN_Mingguan_U64_LR0.001_K3.h5', compile=False)
    model_bulanan = load_model('Tuned_TCN_Bulanan_U128_LR0.001_K3.h5', compile=False)
    return model_harian, model_mingguan, model_bulanan

try:
    model_h, model_m, model_b = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Fungsi Prediksi
def predict_future(model, full_data, lookback=60):
    scaler = RobustScaler()
    scaled_all = scaler.fit_transform(full_data.reshape(-1, 1))
    
    # Pastikan data cukup untuk lookback
    if len(scaled_all) < lookback:
        return None
        
    last_sequence = scaled_all[-lookback:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0][0]

# 4. Sidebar Konfigurasi (Agar lookback_val terdefinisi)
with st.sidebar:
    st.write("### Konfigurasi Parameter")
    lookback_val = st.slider("Jendela Lookback (Harian)", 5, 100, 60)
    st.info("Lookback harus sesuai dengan jendela waktu saat training (default: 60).")

# 5. Main UI & Data Scraping
st.subheader("Analisis Harga Real-time (BBCA.JK)")
# Ambil data sedikit lebih banyak untuk memastikan grafik terisi
df = yf.download("BBCA.JK", period='1mo', interval='1d') 

if not df.empty:
    # 1. Penanganan Kolom (Pastikan mengambil data Close terbaru)
    # yfinance terbaru sering menghasilkan MultiIndex, kita ambil level terendah
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0]
    else:
        close_data = df['Close']

    # 2. Menampilkan Grafik (Pastikan menggunakan format yang didukung Streamlit)
    st.line_chart(close_data)
    
    # 3. Output Real-time (Menampilkan Harga Saat Ini)
    last_price = close_data.iloc[-1]
    last_date = df.index[-1].date()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Harga Terakhir (Market Close)", value=f"Rp {last_price:,.2f}", delta=f"Tanggal: {last_date}")

    # 4. Tabel Data (Terbaru di atas)
    with st.expander("Lihat Tabel Data Historis"):
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

    # 5. Tombol Prediksi
    if st.button('Mulai Prediksi Harga Selanjutnya'):
        with st.spinner('Menganalisis pola...'):
            latest_prices = close_data.values
            hasil = predict_future(model_h, latest_prices, lookback=lookback_val)
            
            if hasil:
                with col2:
                    st.metric(label="Hasil Prediksi Selanjutnya", value=f"Rp {hasil:,.2f}")
                
                st.success(f"### Analisis Selesai")
                st.write(f"Berdasarkan data terakhir tanggal **{last_date}**, model memprediksi harga akan bergerak ke arah **Rp {hasil:,.2f}** pada hari bursa berikutnya.")
            else:
                st.error(f"Data tidak cukup! Butuh {lookback_val} data, tersedia {len(latest_prices)}.")
else:
    st.warning("Gagal mengambil data. Coba refresh halaman.")
