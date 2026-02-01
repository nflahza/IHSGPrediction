import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(page_title="IHSG Predictor 2026", layout="wide")

st.title("📈 Dashboard Prediksi IHSG Indonesia")
st.write(f"Update Terakhir: {datetime.now().strftime('%Y-%m-%d')}")

# 1. Load Model dan Scaler
@st.cache_resource # Agar tidak reload terus menerus
def load_assets():
    model = load_model('model_ihsg.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 2. Sidebar untuk Input
st.sidebar.header("Pengaturan Prediksi")
days_to_predict = st.sidebar.slider("Jumlah hari prediksi ke depan:", 7, 365, 30)

# 3. Download Data Terbaru
ticker = "^JKSE"
data = yf.download(ticker, start="2023-01-01", end=datetime.now().strftime('%Y-%m-%d'))
data.columns = data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else data.columns

# 4. Tampilkan Data Historis
tahun = 4
HARI_DALAM_TAHUN = 365

st.subheader("Data Historis Terbaru")
st.line_chart(data['Close'].tail(HARI_DALAM_TAHUN*tahun), color="##81c995", width="content")

# 5. Logika Prediksi Masa Depan
if st.button("Jalankan Prediksi Masa Depan"):
    # Preprocessing 60 hari terakhir
    last_60_days = data['Close'].tail(60).values.reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    
    current_batch = last_60_days_scaled.reshape((1, 60, 1))
    future_predictions = []

    with st.spinner('Menghitung prediksi...'):
        for i in range(days_to_predict):
            pred = model.predict(current_batch, verbose=0)
            future_predictions.append(pred[0])
            current_batch = np.append(current_batch[:, 1:, :], [pred], axis=1)

    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Buat DataFrame hasil
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1, freq='B')[1:]
    df_pred = pd.DataFrame(future_predictions, index=future_dates, columns=['Prediksi'])

    # 6. Visualisasi Hasil
    st.subheader(f"Hasil Proyeksi {days_to_predict} Hari ke Depan")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Close'].tail(50), label='Historis')
    ax.plot(df_pred['Prediksi'], label='Prediksi', color='red', linestyle='--')
    ax.legend()
    st.pyplot(fig)
    
    st.write(df_pred)