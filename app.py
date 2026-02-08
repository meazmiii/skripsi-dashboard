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
st.subheader("Data Saham Terkini (BBCA.JK)")
df = yf.download("BBCA.JK", period='2y')

if not df.empty:
    # Solusi KeyError Adj Close (Tetap dipertahankan)
    if 'Adj Close' in df.columns:
        close_data = df['Adj Close']
    elif ('Adj Close', 'BBCA.JK') in df.columns:
        close_data = df[('Adj Close', 'BBCA.JK')]
    else:
        close_data = df['Close']
        
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    # Grafik Harga (Visualisasi 100 hari terakhir)
    st.line_chart(close_data.tail(100))
    
    # Tabel Data (Terbaru di atas)
    with st.expander("Lihat Tabel Data Historis"):
        # Kita balik urutannya agar tanggal terbaru ada di baris pertama
        df_sorted = df.sort_index(ascending=False)
        # Menampilkan 10 data teratas
        st.dataframe(df_sorted.head(10), use_container_width=True)

    if st.button('Mulai Prediksi Harga Harian'):
        with st.spinner('Menganalisis pola dengan Tuned LSTM...'):
            latest_prices = close_data.values
            hasil = predict_future(model_h, latest_prices, lookback=lookback_val)
            
            if hasil:
                # Menampilkan hasil dengan format Rupiah yang rapi
                st.success(f"### Prediksi harga untuk hari bursa berikutnya: Rp {hasil:,.2f}")
                st.info(f"💡 **Informasi:** Prediksi ini menggunakan data terakhir pada tanggal {df.index[-1].date()}.")
            else:
                st.error(f"Data tidak cukup! Kamu butuh minimal {lookback_val} hari data, sedangkan data tersedia hanya {len(latest_prices)} hari.")
else:
    st.warning("Gagal mengambil data dari Yahoo Finance. Periksa koneksi internet atau simbol saham.")
