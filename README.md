# STUDI KOMPARASI MODEL LSTM DAN TCN DALAM PREDIKSI SAHAM BBCA PADA TIGA TIME FRAME : HARIAN, MINGGUAN, BULANAN


Penelitian ini merupakan studi mendalam mengenai perbandingan performa antara dua arsitektur Deep Learning, yaitu Long Short-Term Memory (LSTM) dan Temporal Convolutional Network (TCN). Fokus utama penelitian ini adalah mengevaluasi kemampuan kedua model dalam menangkap dependensi temporal dan pola urutan pada data deret waktu yang kompleks dengan tingkat volatilitas berbeda.

## Urgensi Penelitian

Urgensi penelitian ini terletak pada tantangan pemodelan temporal untuk menangkap dependensi urutan jangka pendek maupun panjang pada data dengan volatilitas tinggi. Hasil penelitian ini diharapkan dapat memberikan panduan akurat bagi investor dalam menentukan model prediksi yang paling optimal berdasarkan frekuensi data spesifik.

## Objek dan Subjek Penelitian

Objek Utama: Analisis komparatif kinerja arsitektur LSTM (Sequential Memory) vs TCN (Dilated Convolution).

Subjek Uji (Instrumen): Harga saham historis PT Bank Central Asia Tbk. (BBCA.JK) periode 2004 hingga 2026.

Atribut Data: Adjusted Close Price untuk menjamin akurasi data terhadap aksi korporasi seperti stock split dan dividen.

Timeframe: Evaluasi dilakukan pada tiga skala waktu yaitu Harian, Mingguan, dan Bulanan.

## Metodologi Teknis

Akuisisi Data: Mengambil data sekunder melalui API Yahoo Finance menggunakan library yfinance.

Data Cleaning: Penanganan data invalid pada kolom volume perdagangan untuk menjaga integritas model.

Normalisasi: Menggunakan RobustScaler untuk mentransformasi rentang harga dari Rp97 hingga Rp10.500 menjadi skala desimal yang tahan terhadap outliers.

Sliding Window: Penerapan lookback period sebesar 60 untuk harian, 24 untuk mingguan, dan 12 untuk bulanan.

Overlap Windowing: Teknik menyambung rantai informasi antar sesi scrapping agar prediksi data riil tetap memiliki konteks historis.

Optimasi Model: Implementasi Grid Search untuk menemukan kombinasi hyperparameter terbaik seperti learning rate, hidden units, dan filters.

## Hasil Analisis Akhir

Berdasarkan pengujian komprehensif, ditemukan bahwa karakteristik timeframe sangat mempengaruhi performa masing-masing arsitektur:

1. Timeframe Harian (Jangka Pendek)

Model Terbaik: Tuned LSTM.


Performa: Skor R2 mencapai 0.8679 dengan MAPE terkecil sebesar 2.68%.

Temuan: LSTM sangat unggul pada data yang rapat dan detail dengan selisih harga absolut (MAE) hanya Rp123,39 terhadap harga pasar.

2. Timeframe Mingguan (Jangka Menengah)

Model Terbaik: LSTM.

Performa: Rata-rata MAPE sebesar 4.92% dan MAE sebesar 362.50.

Temuan: LSTM tetap mendominasi dengan selisih harga prediksi dalam Rupiah yang secara konsisten lebih rendah dibandingkan TCN pada skala ini.

3. Timeframe Bulanan (Jangka Panjang)

Model Terbaik: Tuned TCN.

Performa: Menunjukkan stabilitas yang lebih baik pada data renggang dengan skor R2 yang tetap berada pada zona positif.

Temuan: TCN mampu menangkap tren jangka panjang secara lebih efisien dengan rata-rata RMSE sebesar 539,23 saat performa LSTM menurun signifikan hingga 1184,90.

## Teknologi yang Digunakan

Pemrograman: Python.

Deep Learning: TensorFlow dan Keras.

Dashboard: Streamlit untuk visualisasi hasil prediksi secara interaktif.

## Identitas Peneliti

Nama: Azmi Aziz

NIM: 22.11.4903

Instansi: Universitas AMIKOM Yogyakarta
Program Studi: Informatika
