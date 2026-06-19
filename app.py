import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from tf_keras.models import load_model

st.set_page_config(page_title="AMZN Predictor", page_icon="📈", layout="centered")

PREDICTION_DAYS = 60
N_FUTURE = 14

@st.cache_resource
def load_artifacts():
    model = load_model("amazon_lstm_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data(ttl=3600)
def fetch_data():
    df = yf.Ticker("AMZN").history(period="6mo")
    return df[["Close"]].dropna()

def predict_future(model, scaler, df):
    data = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    seq = scaled[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 1)
    preds = []
    for _ in range(N_FUTURE):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    dates = [last_date + timedelta(days=i+1) for i in range(N_FUTURE)]
    return dates, preds

st.title("📈 Amazon - Prediction 2 semaines")
st.caption("Modele LSTM charge depuis le repo")

with st.spinner("Chargement..."):
    try:
        model, scaler = load_artifacts()
        df = fetch_data()
    except Exception as e:
        st.error(f"Erreur : {e}")
        st.stop()

dates, preds = predict_future(model, scaler, df)

last_price = df["Close"].iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Cours actuel", f"${last_price:.2f}")
col2.metric("Prediction J+7", f"${preds[6]:.2f}", f"{((preds[6]-last_price)/last_price*100):+.2f}%")
col3.metric("Prediction J+14", f"${preds[13]:.2f}", f"{((preds[13]-last_price)/last_price*100):+.2f}%")

st.divider()

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

hist_dates = df.index[-30:]
hist_vals = df["Close"].values[-30:]
ax.plot(hist_dates, hist_vals, color="#a0a0a0", linewidth=1.5, label="Historique (30j)")
ax.plot(dates, preds, color="#f6a623", linewidth=2, linestyle="--", marker="o", markersize=4, label="Prediction 14j")
ax.axvline(x=df.index[-1], color="#555", linestyle=":", linewidth=1)

ax.set_facecolor("#0e1117")
ax.tick_params(colors="#a0a0a0")
ax.set_title("Prediction cours AMZN - 14 prochains jours", color="white")
ax.set_ylabel("Prix (USD)", color="#a0a0a0")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax.grid(alpha=0.1, color="white")
plt.xticks(rotation=30, color="#a0a0a0")
fig.tight_layout()
st.pyplot(fig)

st.subheader("Detail des predictions")
df_out = pd.DataFrame({
    "Date": [d.strftime("%d/%m/%Y") for d in dates],
    "Jour": [f"J+{i+1}" for i in range(N_FUTURE)],
    "Prix predit (USD)": [f"${p:.2f}" for p in preds],
    "Variation": [f"{((p-last_price)/last_price*100):+.2f}%" for p in preds]
})
st.dataframe(df_out, use_container_width=True, hide_index=True)
