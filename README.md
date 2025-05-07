import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import streamlit as st

# === STREAMLIT CONFIGURATION ===
st.set_page_config(page_title="EUR/CHF LSTM Forecast", layout="wide")
st.title("EUR/CHF Forecast with LSTM (Including SNB Interventions)")
st.markdown("This project forecasts EUR/CHF exchange rate movements using an LSTM model, incorporating manually tagged SNB interventions. Built with TensorFlow, Plotly, and Streamlit.")

# === DATA LOADING ===
@st.cache_data
def load_data():
    df = yf.download("EURCHF=X", start="2010-01-01", end="2024-12-31")[["Close"]].dropna()
    df = df.rename(columns={"Close": "EURCHF"})
    df.index = pd.to_datetime(df.index)

    # Synthetic SNB intervention markers
    interventions = pd.DataFrame({
        "Date": pd.to_datetime([
            "2011-08-03", "2011-09-06", "2015-01-15", "2020-03-19", "2022-06-16"
        ]),
        "Intervention": 1
    })
    df["Intervention"] = 0
    df.loc[df.index.isin(interventions["Date"]), "Intervention"] = 1

    return df

df = load_data()

# === DATA PREPROCESSING ===
scaler = MinMaxScaler()
df[["EURCHF"]] = scaler.fit_transform(df[["EURCHF"]])
window_size = 30
X, y, y_dates = [], [], []

for i in range(window_size, len(df)):
    features = df.iloc[i - window_size:i][["EURCHF", "Intervention"]].values
    X.append(features)
    y.append(df.iloc[i]["EURCHF"])
    y_dates.append(df.index[i])  # Capture target date

X, y, y_dates = np.array(X), np.array(y), np.array(y_dates)

# === TRAIN/TEST SPLIT ===
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = y_dates[split:]

# === LSTM MODEL ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, 2)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

# === PREDICTION & EVALUATION ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# === RESULTS DISPLAY ===
st.subheader("📈 Model Performance")
st.markdown(f"**MAE**: `{mae:.4f}` &nbsp;&nbsp;&nbsp; **R²**: `{r2:.4f}`")

# === VISUALIZATION ===
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_rescaled = scaler.inverse_transform(y_pred).flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_rescaled, name="Actual"))
fig.add_trace(go.Scatter(x=test_dates, y=y_pred_rescaled, name="Predicted"))
fig.update_layout(title="EUR/CHF Forecast with LSTM", xaxis_title="Date", yaxis_title="EUR/CHF")
fig.show()
