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

# --- CSS CUSTOM: Mengembalikan format Tanggal & Jam besar ---
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
    return load_model(path, compile=False)

@st.cache_data(ttl=None)
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

# --- HEADER: Format Sejajar & Font Besar Sesuai Request ---
st.title("🚀 Dashboard Analisis Saham BBCA (LSTM & TCN)")
st.write("") 

col_jam, col_tgl = st.columns([1, 1])
with col_jam:
    st.markdown(f"## **Waktu Real-time:** <span style='font-size: 38px;'>`{now_jkt.strftime('%H:%M:%S')}`</span> **WIB**", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data Aktual"):
        st.cache_data.clear()
        st.rerun()

with col_tgl:
    st.markdown(f"<div style='text-align: right;'><h2><b>Tanggal:</b> <span style='font-size: 38px;'><code>{now_jkt.strftime('%d-%m-%Y')}</code></span></h2></div>", unsafe_allow_html=True)

df_all = get_data_manual()

# 3. Struktur Dashboard Utama
if not df_all.empty:
 # 1. Tambahkan tab_comp di urutan pertama
    tab_comp, tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Prediksi", "📅 Harian", "🗓️ Mingguan", "📊 Bulanan"])

    with tab_comp:
        st.subheader("🚀 Komparasi Prediksi Harga: 12 Skenario Model")
        
        # --- CSS UNTUK TAMPILAN KARTU ---
        st.markdown("""
            <style>
                .card-container {
                    background-color: #1E1E1E;
                    border-radius: 15px;
                    padding: 20px;
                    border: 1px solid #333;
                    box-shadow: 2px 4px 10px rgba(0,0,0,0.3);
                    margin-bottom: 20px;
                }
                .card-title {
                    font-size: 20px;
                    font-weight: bold;
                    color: #00AAFF;
                    border-bottom: 1px solid #444;
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                }
                .model-box {
                    display: flex;
                    justify-content: space-between;
                    background: rgba(255,255,255,0.05);
                    padding: 10px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                }
                .price-text {
                    color: #00FFCC;
                    font-weight: bold;
                    font-family: monospace;
                }
            </style>
        """, unsafe_allow_html=True)

        # Ambil data input
        close_h = df_all['Close'].dropna().values
        close_w = df_all['Close'].resample('W-MON').last().dropna().values
        close_m = df_all['Close'].resample('ME').last().dropna().values

        # Mapping Config
        tfs = {
            "HARIAN": {"data": close_h, "lb": 60, "icon": "📅", "models": [
                ("LSTM Standar", "models/baseline/Baseline_LSTM_Harian.h5"),
                ("TCN Standar", "models/baseline/Baseline_TCN_Harian.h5"),
                ("LSTM Khusus", "models/tuned/Tuned_LSTM_Harian_U64_LR0.001_KN.h5"),
                ("TCN Khusus", "models/tuned/Tuned_TCN_Harian_U128_LR0.001_K2.h5")
            ]},
            "MINGGUAN": {"data": close_w, "lb": 24, "icon": "🗓️", "models": [
                ("LSTM Standar", "models/baseline/Baseline_LSTM_Mingguan.h5"),
                ("TCN Standar", "models/baseline/Baseline_TCN_Mingguan.h5"),
                ("LSTM Khusus", "models/tuned/Tuned_LSTM_Mingguan_U64_LR0.001_KN.h5"),
                ("TCN Khusus", "models/tuned/Tuned_TCN_Mingguan_U64_LR0.001_K3.h5")
            ]},
            "BULANAN": {"data": close_m, "lb": 12, "icon": "📊", "models": [
                ("LSTM Standar", "models/baseline/Baseline_LSTM_Bulanan.h5"),
                ("TCN Standar", "models/baseline/Baseline_TCN_Bulanan.h5"),
                ("LSTM Khusus", "models/tuned/Tuned_LSTM_Bulanan_U128_LR0.0001_KN.h5"),
                ("TCN Khusus", "models/tuned/Tuned_TCN_Bulanan_U128_LR0.001_K3.h5")
            ]}
        }

        # Render Kolom
        cols = st.columns(3)
        for i, (name, config) in enumerate(tfs.items()):
            with cols[i]:
                # Header Kartu (HTML)
                st.markdown(f"""<div class="card-container"><div class="card-title">{config['icon']} {name}</div>""", unsafe_allow_html=True)
                
                # Isi Model (Looping Python + Markdown untuk tampilan)
                for m_name, m_path in config['models']:
                    try:
                        m_obj = get_model(m_path)
                        val = predict_stock(m_obj, config['data'], config['lb'])
                        
                        # Tampilkan tiap baris model menggunakan container HTML
                        st.markdown(f"""
                            <div class="model-box">
                                <span style="color:#AAA;">{m_name}</span>
                                <span class="price-text">Rp {val:,.2f}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error loading {m_name}")
                
                st.markdown("</div>", unsafe_allow_html=True) # Tutup card-container

    # --- TAB 1: HARIAN (Lookback: 60) ---
    with tab1:
        st.subheader("Analisis Perbandingan & Prediksi Harian")
        close_s = df_all['Close'].dropna()
        last_p = float(close_s.iloc[-1])
        
        # MENAMPILKAN HARGA AKTUAL LANGSUNG
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Harga Aktual Terakhir", f"Rp {last_p:,.2f}")
        
        # RADIO MODEL (BREAKDOWN)
        model_h = st.radio("Pilih Model Harian:", ["LSTM Standar", "TCN Standar", "LSTM Khusus", "TCN Khusus"], index=None, horizontal=True)
        
        if model_h:
            paths = {
                "LSTM Standar": "models/baseline/Baseline_LSTM_Harian.h5",
                "TCN Standar": "models/baseline/Baseline_TCN_Harian.h5",
                "LSTM Khusus": "models/tuned/Tuned_LSTM_Harian_U64_LR0.001_KN.h5",
                "TCN Khusus": "models/tuned/Tuned_TCN_Harian_U128_LR0.001_K2.h5"
            }
            curr_model = get_model(paths[model_h])
            pred_p = predict_stock(curr_model, close_s.values, 60)
            
            # Tampilkan Prediksi di Samping Harga Aktual
            with c2:
                st.metric(f"Prediksi {model_h}", f"Rp {pred_p:,.2f}")
            
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
            st.info("💡 Pilih salah satu model di atas untuk memunculkan harga prediksi dan tabel akurasi.")

        with st.expander("Lihat Data Historis Harian Lengkap"):
            st.dataframe(df_all.sort_index(ascending=False), use_container_width=True)

    # --- TAB 2: MINGGUAN (Lookback: 24) ---
    with tab2:
        st.subheader("Analisis Perbandingan & Prediksi Mingguan")
        
        # Perbaikan: Menggunakan .agg untuk mendapatkan OHLCV lengkap
        df_w_full = df_all.resample('W-MON').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        last_p_w = float(df_w_full['Close'].iloc[-1])
        
        cw1, cw2 = st.columns(2)
        with cw1: st.metric("Harga Aktual Minggu Ini", f"Rp {last_p_w:,.2f}")
            
        model_w = st.radio("Pilih Model Mingguan:", ["LSTM Standar", "TCN Standar", "LSTM Khusus", "TCN Khusus"], index=None, horizontal=True)
        
        if model_w:
            paths_w = {
                "LSTM Standar": "models/baseline/Baseline_LSTM_Mingguan.h5",
                "TCN Standar": "models/baseline/Baseline_TCN_Mingguan.h5",
                "LSTM Khusus": "models/tuned/Tuned_LSTM_Mingguan_U64_LR0.001_KN.h5",
                "TCN Khusus": "models/tuned/Tuned_TCN_Mingguan_U64_LR0.001_K3.h5"
            }
            curr_model_w = get_model(paths_w[model_w])
            # Prediksi tetap pakai kolom Close saja
            pred_p_w = predict_stock(curr_model_w, df_w_full['Close'].values, 24)
            
            with cw2: st.metric(f"Prediksi {model_w}", f"Rp {pred_p_w:,.2f}")
            
            # Tabel Akurasi (tetap 5 baris terakhir)
            st.write(f"### 🕒 Historis 5 Minggu Terakhir ({model_w})")
            hist_w = []
            for i in range(1, 6):
                t_idx = -i
                act_w = df_w_full['Close'].iloc[t_idx]
                p_val_w = predict_stock(curr_model_w, df_w_full['Close'].iloc[:t_idx].values, 24)
                hist_w.append({"Tanggal": df_w_full.index[t_idx].date(), "Harga Aktual": f"Rp {act_w:,.2f}", "Prediksi": f"Rp {p_val_w:,.2f}", "Selisih": f"{abs(act_w - p_val_w):,.2f}"})
            st.table(pd.DataFrame(hist_w))
        else:
            st.info("💡 Pilih salah satu model untuk melihat perbandingan mingguan.")
    
        with st.expander("Lihat Data Historis Mingguan Lengkap"):
            # Tampilkan DataFrame lengkap yang sudah di-resample
            st.dataframe(df_w_full.sort_index(ascending=False), use_container_width=True)

    # --- TAB 3: BULANAN (Lookback: 12) ---
    with tab3:
        st.subheader("Analisis Perbandingan & Prediksi Bulanan")
        
        # Perbaikan: Menggunakan .agg untuk mendapatkan OHLCV lengkap
        df_m_full = df_all.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        last_p_m = float(df_m_full['Close'].iloc[-1])
        
        cm1, cm2 = st.columns(2)
        with cm1: st.metric("Harga Aktual Bulan Ini", f"Rp {last_p_m:,.2f}")
            
        model_m = st.radio("Pilih Model Bulanan:", ["LSTM Standar", "TCN Standar", "LSTM Khusus", "TCN Khusus"], index=None, horizontal=True)
        
        if model_m:
            paths_m = {
                "LSTM Standar": "models/baseline/Baseline_LSTM_Bulanan.h5",
                "TCN Standar": "models/baseline/Baseline_TCN_Bulanan.h5",
                "LSTM Khusus": "models/tuned/Tuned_LSTM_Bulanan_U128_LR0.0001_KN.h5",
                "TCN Khusus": "models/tuned/Tuned_TCN_Bulanan_U128_LR0.001_K3.h5"
            }
            curr_model_m = get_model(paths_m[model_m])
            pred_p_m = predict_stock(curr_model_m, df_m_full['Close'].values, 12)
            
            with cm2: st.metric(f"Prediksi {model_m}", f"Rp {pred_p_m:,.2f}")
            
            st.write(f"### 🕒 Historis 5 Bulan Terakhir ({model_m})")
            hist_m = []
            for i in range(1, 6):
                t_idx = -i
                act_m = df_m_full['Close'].iloc[t_idx]
                p_val_m = predict_stock(curr_model_m, df_m_full['Close'].iloc[:t_idx].values, 12)
                hist_m.append({"Bulan": df_m_full.index[t_idx].strftime('%B %Y'), "Harga Aktual": f"Rp {act_m:,.2f}", "Prediksi": f"Rp {p_val_m:,.2f}", "Selisih": f"{abs(act_m - p_val_m):,.2f}"})
            st.table(pd.DataFrame(hist_m))
        else:
            st.info("💡 Pilih salah satu model untuk melihat perbandingan bulanan.")
    
        with st.expander("Lihat Data Historis Bulanan Lengkap"):
            # Tampilkan DataFrame lengkap yang sudah di-resample
            st.dataframe(df_m_full.sort_index(ascending=False), use_container_width=True)

# --- Copyright ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(f"""
    <div style='text-align: center;'>
        <b>AZMI AZIZ | 22.11.4903</b><br>
        © 2026 Universitas AMIKOM Yogyakarta
    </div>
    <div style='text-align: center; margin-top: 10px;'>
        <a href='https://www.instagram.com/_azmiazzz/?hl=id' target='_blank' style='color: #00AAFF; text-decoration: none; font-weight: bold; font-size: 16px;'>
            📸 Instagram: @_azmiazzz
        </a>
    </div>
""", unsafe_allow_html=True)








