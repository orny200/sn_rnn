import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

try:
    from tf_keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

st.set_page_config(page_title="Amazon Stock Predictor", page_icon="📈", layout="wide")

PREDICTION_DAYS = 60

@st.cache_resource
def load_artifacts():
    model = load_model("amazon_lstm_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data(ttl=3600)
def fetch_data(period="2y"):
    ticker = yf.Ticker("AMZN")
    df = ticker.history(period=period)
    return df[["Close"]].dropna()

def predict_on_history(model, scaler, df):
    data = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    train_size = int(len(scaled) * 0.8)
    test_data = scaled[train_size - PREDICTION_DAYS:, :]
    X_test = []
    for i in range(PREDICTION_DAYS, len(test_data)):
        X_test.append(test_data[i - PREDICTION_DAYS:i, 0])
    X_test = np.array(X_test).reshape(-1, PREDICTION_DAYS, 1)
    preds = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual_dates = df.index[train_size:]
    actual_vals = data[train_size:]
    return actual_dates, actual_vals.flatten(), preds.flatten()

def predict_future(model, scaler, df, n_days=30):
    data = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    seq = scaled[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 1)
    future_preds = []
    for _ in range(n_days):
        pred = model.predict(seq, verbose=0)[0, 0]
        future_preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    return future_dates, future_preds

st.title("📈 Amazon Stock Price Predictor")
st.caption("Modele LSTM entraine sur l'historique complet du cours AMZN")

with st.sidebar:
    st.header("Parametres")
    period = st.selectbox("Historique a charger", options=["6mo", "1y", "2y", "5y", "max"], index=2)
    n_future = st.slider("Jours a predire dans le futur", min_value=5, max_value=90, value=30, step=5)
    show_future = st.checkbox("Afficher la prediction future", value=True)
    show_metrics = st.checkbox("Afficher les metriques", value=True)
    st.divider()
    st.info("Les predictions futures sont basees sur les 60 derniers jours de cours.")

with st.spinner("Chargement du modele et des donnees..."):
    try:
        model, scaler = load_artifacts()
        df = fetch_data(period)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

last_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]
change = last_price - prev_price
change_pct = change / prev_price * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dernier cours", f"${last_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
col2.metric("Plus haut (periode)", f"${df['Close'].max():.2f}")
col3.metric("Plus bas (periode)", f"${df['Close'].min():.2f}")
col4.metric("Donnees chargees", f"{len(df):,} jours")

st.divider()

with st.spinner("Generation des predictions..."):
    actual_dates, actual_vals, hist_preds = predict_on_history(model, scaler, df)
    if show_future:
        future_dates, future_preds = predict_future(model, scaler, df, n_days=n_future)

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

ax.plot(df.index, df["Close"].values, color="#a0a0a0", linewidth=1, label="Cours reel (complet)", alpha=0.5)
ax.plot(actual_dates, hist_preds, color="#00d084", linewidth=1.5, label="Prediction LSTM (test)")

if show_future:
    ax.plot(future_dates, future_preds, color="#f6a623", linewidth=2, linestyle="--", label=f"Prediction future ({n_future}j)")
    ax.axvline(x=df.index[-1], color="#f6a623", linestyle=":", alpha=0.5, linewidth=1)

ax.set_title("AMZN - Prediction cours de bourse (LSTM)", color="white", fontsize=14, pad=12)
ax.set_xlabel("Date", color="#a0a0a0")
ax.set_ylabel("Prix (USD)", color="#a0a0a0")
ax.tick_params(colors="#a0a0a0")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=30)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax.grid(alpha=0.1, color="white")
fig.tight_layout()
st.pyplot(fig)

if show_metrics:
    st.subheader("Performance du modele (donnees de test)")
    mae = np.mean(np.abs(actual_vals - hist_preds))
    rmse = np.sqrt(np.mean((actual_vals - hist_preds) ** 2))
    mape = np.mean(np.abs((actual_vals - hist_preds) / actual_vals)) * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${mae:.2f}")
    m2.metric("RMSE", f"${rmse:.2f}")
    m3.metric("MAPE", f"{mape:.2f}%")

if show_future:
    st.subheader(f"Predictions - {n_future} prochains jours")
    df_future = pd.DataFrame({
        "Date": [d.strftime("%d/%m/%Y") for d in future_dates],
        "Prix predit (USD)": [f"${p:.2f}" for p in future_preds],
        "Variation vs aujourd'hui": [f"{((p - last_price) / last_price * 100):+.2f}%" for p in future_preds]
    })
    st.dataframe(df_future, use_container_width=True, hide_index=True)

st.divider()
st.caption("Avertissement : Ce modele est developpe a des fins academiques. Les predictions ne constituent pas des conseils financiers.")
