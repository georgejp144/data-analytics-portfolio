from dotenv import load_dotenv
from math import sqrt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ML + preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Technical indicators
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Alpaca Data API
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Deep Learning (LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib

# ======================================================
# 1. LOAD & PREP DATA
# ======================================================
load_dotenv()

ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")

client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)

end = datetime.now(timezone.utc)
start = end - timedelta(days=365 * 3)

req = StockBarsRequest(
    symbol_or_symbols=["QQQ"],
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    adjustment='all',
    feed=DataFeed.IEX
)

bars = client.get_stock_bars(req).df
df = (
    bars.xs("QQQ", level="symbol")
        .reset_index()
        .rename(columns={
            "timestamp": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        .sort_values("date")
        .reset_index(drop=True)
)

# ======================================================
# 2. FEATURES + TARGET
# ======================================================
df["ret"] = np.log(df["Close"]).diff()

WINDOW = 14
TRADING_DAYS = 252
df["rv_14_back"] = (
    df["ret"].rolling(WINDOW).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    * np.sqrt(TRADING_DAYS)
)

df["rv_1"] = df["rv_14_back"].shift(1)
df["rv_5"] = (df["ret"].rolling(5).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True) * np.sqrt(TRADING_DAYS)).shift(1)
df["rv_22"] = (df["ret"].rolling(22).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True) * np.sqrt(TRADING_DAYS)).shift(1)
df["rv_5_22_ratio"] = df["rv_5"] / df["rv_22"]

df["rsi_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
df["bb_bandwidth"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
df["atr_14"] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

df["ret_std_5"] = df["ret"].rolling(5).std() * np.sqrt(TRADING_DAYS)
df["ret_std_22"] = df["ret"].rolling(22).std() * np.sqrt(TRADING_DAYS)
df["vol_z_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

df["rv_14_forward"] = df["rv_14_back"].shift(-WINDOW)

FEATURES = [
    "rv_1", "rv_5", "rv_22", "rv_5_22_ratio",
    "rsi_14", "bb_bandwidth", "atr_14",
    "ret_std_5", "ret_std_22", "vol_z_20"
]

df_train = df.dropna(subset=FEATURES + ["rv_14_forward"]).copy()
df_latest = df.dropna(subset=FEATURES)

X = df_train[FEATURES].values
y = df_train["rv_14_forward"].values

split_idx = int(len(df_train) * 0.8)
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)

# ======================================================
# 3. BUILD 22-DAY SEQUENCE DATASETS
# ======================================================
def make_sequence_dataset(features, target, window=22):
    X_seq, y_seq = [], []
    for i in range(window, len(features)):
        X_seq.append(features[i-window:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)

WINDOW_SEQ = 22
X_train, y_train = make_sequence_dataset(X_train_s, y_train_raw, window=WINDOW_SEQ)
X_test, y_test = make_sequence_dataset(X_test_s, y_test_raw, window=WINDOW_SEQ)

# ======================================================
# 4. TRAIN LSTM
# ======================================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SEQ, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mae")

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("\nTraining LSTM with 22-day sequences...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=16,
    callbacks=[es],
    verbose=1
)

# ======================================================
# 5. EVALUATE
# ======================================================
pred_test = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, pred_test)
rmse = sqrt(mean_squared_error(y_test, pred_test))
relative_mae = (mae / np.mean(y_test)) * 100

print(f"\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Relative MAE: {relative_mae:.2f}%")

# ======================================================
# 6. FORECAST NEXT 14D RV
# ======================================================
latest_block = scaler.transform(df_latest[FEATURES].values)[-WINDOW_SEQ:]
latest_block = latest_block.reshape(1, WINDOW_SEQ, len(FEATURES))

rv_next14_pred = model.predict(latest_block)[0][0]

print(f"\nPredicted NEXT 14-Day Realised Vol (annualised): {rv_next14_pred:.3f}")
print(f"Latest data date: {df_latest['date'].iloc[-1].date()}")

# ======================================================
# 7. SAVE ARTIFACTS
# ======================================================
model.save("rv_har_lstm_seq_model.h5")
joblib.dump(scaler, "rv_feature_scaler.joblib")
df_train.to_csv("rv_training_data.csv", index=False)
