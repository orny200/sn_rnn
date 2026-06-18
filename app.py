import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ─── Configuration ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Amazon Stock Predictor",
    page_icon="📈",
    layout="wide"
)

PREDICTION_DAYS = 60  # doit correspondre à ce qui a servi à l'entraînement

# ─── Chargement des artefacts ────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model = load_model("amazon_lstm_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# ─── Récupération des données ────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_data(period="2y"):
    ticker = yf.Ticker("AMZN")
    df = ticker.history(period=period)
    return df[["Close"]].dropna()

# ─── Prédiction sur données historiques (test split) ─────────────────────────

def predict_on_history(model, scaler, df):
    data = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)

    train_size = int(len(scaled) * 0.8)
    test_data  = scaled[train_size - PREDICTION_DAYS:, :]

    X_test = []
    for i in range(PREDICTION_DAYS, len(test_data)):
        X_test.append(test_data[i - PREDICTION_DAYS:i, 0])

    X_test = np.array(X_test).reshape(-1, PREDICTION_DAYS, 1)
    preds  = scaler.inverse_transform(model.predict(X_test, verbose=0))

    actual_dates = df.index[train_size:]
    actual_vals  = data[train_size:]
    return actual_dates, actual_vals.flatten(), preds.flatten()

# ─── Prédiction des N prochains jours ────────────────────────────────────────

def predict_future(model, scaler, df, n_days=30):
    data   = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    seq    = scaled[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 1)

    future_preds = []
    for _ in range(n_days):
        pred = model.predict(seq, verbose=0)[0, 0]
        future_preds.append(pred)
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

    last_date    = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    return future_dates, future_preds

# ─── Interface ────────────────────────────────────────────────────────────────

st.title("📈 Amazon Stock Price Predictor")
st.caption("Modèle LSTM entraîné sur l'historique complet du cours AMZN")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    period = st.selectbox(
        "Historique à charger",
        options=["6mo", "1y", "2y", "5y", "max"],
        index=2,
        help="Période de données Yahoo Finance"
    )
    n_future = st.slider(
        "Jours à prédire dans le futur",
        min_value=5, max_value=90, value=30, step=5
    )
    show_future = st.checkbox("Afficher la prédiction future", value=True)
    show_metrics = st.checkbox("Afficher les métriques", value=True)
    st.divider()
    st.info("Les prédictions futures sont basées sur les 60 derniers jours de cours.")

# Chargement
with st.spinner("Chargement du modèle et des données..."):
    try:
        model, scaler = load_artifacts()
        df = fetch_data(period)
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e}\n\nAssurez-vous que `amazon_lstm_model.keras` et `scaler.pkl` sont dans le même dossier que `app.py`.")
        st.stop()

# ── Indicateurs clés ──────────────────────────────────────────────────────────

last_price  = df["Close"].iloc[-1]
prev_price  = df["Close"].iloc[-2]
change      = last_price - prev_price
change_pct  = change / prev_price * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dernier cours", f"${last_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
col2.metric("Plus haut (période)", f"${df['Close'].max():.2f}")
col3.metric("Plus bas (période)",  f"${df['Close'].min():.2f}")
col4.metric("Données chargées",    f"{len(df):,} jours")

st.divider()

# ── Prédiction sur historique ─────────────────────────────────────────────────

with st.spinner("Génération des prédictions..."):
    actual_dates, actual_vals, hist_preds = predict_on_history(model, scaler, df)
    if show_future:
        future_dates, future_preds = predict_future(model, scaler, df, n_days=n_future)

# ── Graphique principal ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

# Données réelles
ax.plot(df.index, df["Close"].values, color="#a0a0a0", linewidth=1, label="Cours réel (complet)", alpha=0.5)

# Prédictions test
ax.plot(actual_dates, hist_preds, color="#00d084", linewidth=1.5, label="Prédiction LSTM (test)")

# Prédiction future
if show_future:
    ax.plot(future_dates, future_preds, color="#f6a623", linewidth=2,
            linestyle="--", label=f"Prédiction future ({n_future}j)")
    ax.axvline(x=df.index[-1], color="#f6a623", linestyle=":", alpha=0.5, linewidth=1)

ax.set_title("AMZN — Prédiction cours de bourse (LSTM)", color="white", fontsize=14, pad=12)
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

# ── Métriques de performance ──────────────────────────────────────────────────

if show_metrics:
    st.subheader("📊 Performance du modèle (données de test)")
    mae  = np.mean(np.abs(actual_vals - hist_preds))
    rmse = np.sqrt(np.mean((actual_vals - hist_preds) ** 2))
    mape = np.mean(np.abs((actual_vals - hist_preds) / actual_vals)) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"${mae:.2f}",   help="Mean Absolute Error")
    m2.metric("RMSE", f"${rmse:.2f}",  help="Root Mean Squared Error")
    m3.metric("MAPE", f"{mape:.2f}%",  help="Mean Absolute Percentage Error")

# ── Tableau des prédictions futures ──────────────────────────────────────────

if show_future:
    st.subheader(f"📅 Prédictions — {n_future} prochains jours")
    df_future = pd.DataFrame({
        "Date":              [d.strftime("%d/%m/%Y") for d in future_dates],
        "Prix prédit (USD)": [f"${p:.2f}" for p in future_preds],
        "Variation vs aujourd'hui": [
            f"{((p - last_price) / last_price * 100):+.2f}%" for p in future_preds
        ]
    })
    st.dataframe(df_future, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ **Avertissement** : Ce modèle est développé à des fins académiques. "
    "Les prédictions ne constituent pas des conseils financiers.")
