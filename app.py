import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Dashboard Skripsi BBCA - Azmi Aziz", layout="wide")

# Penentuan Waktu WIB
tz_jkt = pytz.timezone('Asia/Jakarta')
now_jkt = datetime.now(tz_jkt)

# --- CSS CUSTOM UNTUK TABEL DAN TAMPILAN ---
st.markdown("""
    <style>
    .stDataFrame, [data-testid="stTable"] { color: #FFFFFF !important; }
    th { color: #FFFFFF !important; background-color: #31333F !important; }
    td { color: #FFFFFF !important; }
    .streamlit-expanderHeader { color: white !important; background-color: #262730 !important; }
    .big-font { font-size: 38px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. Fungsi Load Model & Data
@st.cache_resource
def get_model(path):
    # Fungsi ini akan memuat model secara dinamis saat dipilih
    return load_model(path, compile=False)

@st.cache_data(ttl=None) # Manual Refresh Only
def get_data_manual():
    ticker = "BBCA.JK"
    try:
        df = yf.download(ticker, period='5y', progress=False)
        if df.empty: raise ValueError("API Limit")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.to_csv("bbca_fallback.csv")
        return df
    except:
        return pd.read_csv("bbca_fallback.csv", index_col=0, parse_dates=True)

def predict_stock(model, data, lookback):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    if len(scaled_data) < lookback: return 0.0
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    prediction_scaled = model.predict(last_sequence, verbose=0)
    return scaler.inverse_transform(prediction_scaled)[0][0]

# --- HEADER & REFRESH CONTROL ---
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")

col_header_1, col_header_2 = st.columns([2, 1])
with col_header_1:
    st.markdown(f"### Waktu Real-time: <span class='big-font'>`{now_jkt.strftime('%H:%M:%S')}`</span> WIB", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data Aktual"):
        st.cache_data.clear()
        st.rerun()

with col_header_2:
    st.markdown(f"<div style='text-align: right;'>### Tanggal: <br><span class='big-font'>`{now_jkt.strftime('%d-%m-%Y')}`</span></div>", unsafe_allow_html=True)

df_all = get_data_manual()

# 3. Struktur Dashboard Utama
if not df_all.empty:
    tab1, tab2, tab3 = st.tabs(["📅 Harian", "🗓️ Mingguan", "📊 Bulanan"])

    # --- TAB 1: HARIAN (Lookback: 60) ---
    with tab1:
        st.subheader("Analisis Multi-Model Harian")
        model_h = st.radio("Pilih Model Harian:", ["LSTM Baseline", "TCN Baseline", "LSTM Tuned", "TCN Tuned"], index=None, horizontal=True, key="h_radio")
        
        if model_h:
            paths = {
                "LSTM Baseline": "models/baseline/Baseline_LSTM_Harian.h5",
                "TCN Baseline": "models/baseline/Baseline_TCN_Harian.h5",
                "LSTM Tuned": "Tuned_LSTM_Harian_U64_LR0.001_KN.h5", # Sesuaikan path file kamu
                "TCN Tuned": "models/tuned/Tuned_TCN_Harian_Best.h5"
            }
            curr_model = get_model(paths[model_h])
            close_s = df_all['Close'].dropna()
            
            # Tampilan Metrik
            last_p = float(close_s.iloc[-1])
            pred_p = predict_stock(curr_model, close_s.values, 60)
            
            c1, c2 = st.columns(2)
            c1.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}")
            c2.metric(f"Prediksi {model_h}", f"Rp {pred_p:,.2f}")
            
            # Tabel Historis
            st.write(f"### 🕒 Historis Akurasi 5 Hari Terakhir ({model_h})")
            hist = []
            for i in range(1, 6):
                t_idx = -i
                act = close_s.iloc[t_idx]
                p_val = predict_stock(curr_model, close_s.iloc[:t_idx].values, 60)
                hist.append({
                    "Tanggal": close_s.index[t_idx].date(),
                    "Harga Aktual": f"Rp {act:,.2f}",
                    f"Prediksi": f"Rp {p_val:,.2f}",
                    "Selisih (Rp)": f"{abs(act - p_val):,.2f}"
                })
            st.table(pd.DataFrame(hist))
        else:
            st.info("Pilih model untuk melihat analisis harian.")

    # --- TAB 2: MINGGUAN (Lookback: 24) ---
    with tab2:
        st.subheader("Analisis Multi-Model Mingguan")
        model_w = st.radio("Pilih Model Mingguan:", ["LSTM Baseline", "TCN Baseline", "LSTM Tuned", "TCN Tuned"], index=None, horizontal=True, key="w_radio")
        
        if model_w:
            df_w = df_all.resample('W-MON').last().dropna()['Close']
            paths_w = {
                "LSTM Baseline": "models/baseline/Baseline_LSTM_Mingguan.h5",
                "TCN Baseline": "models/baseline/Baseline_TCN_Mingguan.h5",
                "LSTM Tuned": "models/tuned/Tuned_LSTM_Mingguan_Best.h5",
                "TCN Tuned": "Tuned_TCN_Mingguan_U64_LR0.001_K3.h5"
            }
            curr_model_w = get_model(paths_w[model_w])
            
            last_p_w = float(df_w.iloc[-1])
            pred_p_w = predict_stock(curr_model_w, df_w.values, 24)
            
            c1w, c2w = st.columns(2)
            c1w.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
            c2w.metric(f"Prediksi {model_w}", f"Rp {pred_p_w:,.2f}")
            
            st.write(f"### 🕒 Historis 5 Minggu Terakhir ({model_w})")
            hist_w = []
            for i in range(1, 6):
                t_idx = -i
                act = df_w.iloc[t_idx]
                p_val = predict_stock(curr_model_w, df_w.iloc[:t_idx].values, 24)
                hist_w.append({"Tanggal": df_w.index[t_idx].date(), "Harga Aktual": f"Rp {act:,.2f}", "Prediksi": f"Rp {p_val:,.2f}", "Selisih": f"{abs(act - p_val):,.2f}"})
            st.table(pd.DataFrame(hist_w))
        else:
            st.info("Pilih model untuk melihat analisis mingguan.")

    # --- TAB 3: BULANAN (Lookback: 12) ---
    with tab3:
        st.subheader("Analisis Multi-Model Bulanan")
        model_m = st.radio("Pilih Model Bulanan:", ["LSTM Baseline", "TCN Baseline", "LSTM Tuned", "TCN Tuned"], index=None, horizontal=True, key="m_radio")
        
        if model_m:
            df_m = df_all.resample('ME').last().dropna()['Close']
            paths_m = {
                "LSTM Baseline": "models/baseline/Baseline_LSTM_Bulanan.h5",
                "TCN Baseline": "models/baseline/Baseline_TCN_Bulanan.h5",
                "LSTM Tuned": "models/tuned/Tuned_LSTM_Bulanan_Best.h5",
                "TCN Tuned": "Tuned_TCN_Bulanan_U128_LR0.001_K3.h5"
            }
            curr_model_m = get_model(paths_m[model_m])
            
            last_p_m = float(df_m.iloc[-1])
            pred_p_m = predict_stock(curr_model_m, df_m.values, 12)
            
            c1m, c2m = st.columns(2)
            c1m.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
            c2m.metric(f"Prediksi {model_m}", f"Rp {pred_p_m:,.2f}")
            
            st.write(f"### 🕒 Historis 5 Bulan Terakhir ({model_m})")
            hist_m = []
            for i in range(1, 6):
                t_idx = -i
                act = df_m.iloc[t_idx]
                p_val = predict_stock(curr_model_m, df_m.iloc[:t_idx].values, 12)
                hist_m.append({"Bulan": df_m.index[t_idx].strftime('%B %Y'), "Harga Aktual": f"Rp {act:,.2f}", "Prediksi": f"Rp {p_val:,.2f}", "Selisih": f"{abs(act - p_val):,.2f}"})
            st.table(pd.DataFrame(hist_m))
        else:
            st.info("Pilih model untuk melihat analisis bulanan.")

# --- FOOTER ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center;'><b>AZMI AZIZ | 22.11.4903</b><br>© 2026 Universitas AMIKOM Yogyakarta</div>", unsafe_allow_html=True)
