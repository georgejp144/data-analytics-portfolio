from dotenv import load_dotenv
from math import sqrt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ML + preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

# Technical indicators
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Alpaca Data API
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# Load env
load_dotenv()

# 1. Authenticates with Alpaca, fetches three years of daily OHLCV bars for QQQ, and tidies them into a clean dataframe for modelling.

ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")

client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)

# Pull ~3 years of daily data - Focused on current market structure

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
df.head()

# 2. Calculates daily log returns and a backward-looking realised volatility over the last 14 days (annualised).
# This is the raw volatility measure you’ll later shift forward to create the forecast target.

df["ret"] = np.log(df["Close"]).diff()

WINDOW = 14
TRADING_DAYS = 252
df["rv_14_back"] = (
    df["ret"].rolling(WINDOW).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    * np.sqrt(TRADING_DAYS)
)

# 3. Creates the HAR lag structure (1-, 5-, and 22-day volatility memory) and adds richer market-state features: RSI, Bollinger Band width, ATR, rolling return volatilities, and volume z-score.
# These capture both memory and nonlinear effects.

# HAR memory features
df["rv_1"]  = df["rv_14_back"].shift(1)
df["rv_5"]  = (df["ret"].rolling(5)
               .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
               * np.sqrt(TRADING_DAYS)).shift(1)
df["rv_22"] = (df["ret"].rolling(22)
               .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
               * np.sqrt(TRADING_DAYS)).shift(1)

df["rv_5_22_ratio"] = df["rv_5"] / df["rv_22"]

# Technical indicators
rsi = RSIIndicator(close=df["Close"], window=14)
df["rsi_14"] = rsi.rsi()

bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
df["bb_bandwidth"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
df["atr_14"] = atr.average_true_range()

# Return volatility and volume stats
df["ret_std_5"]  = df["ret"].rolling(5).std()  * np.sqrt(TRADING_DAYS)
df["ret_std_22"] = df["ret"].rolling(22).std() * np.sqrt(TRADING_DAYS)
df["vol_z_20"]   = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

# 4. Shifts the target 14 days forward to represent future volatility,
# then keeps two datasets — one for training (fully labeled) and one for inference (today’s data row).

# Shift forward to create future target
df["rv_14_forward"] = df["rv_14_back"].shift(-WINDOW)

FEATURES = [
    "rv_1","rv_5","rv_22","rv_5_22_ratio",
    "rsi_14","bb_bandwidth","atr_14",
    "ret_std_5","ret_std_22","vol_z_20"
]

# --- Split cleanly ---
# Training data: complete features + future target
df_train = df.dropna(subset=FEATURES + ["rv_14_forward"]).copy()

# Inference data: most recent row with all features (target may be NaN)
df_latest = df.dropna(subset=FEATURES).iloc[[-1]]

# 5. Splits data chronologically (80 % train / 20 % test) and scales features for stability.

X = df_train[FEATURES].values
y = df_train["rv_14_forward"].values

split_idx = int(len(df_train) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 6. Uses a time-series cross-validation to find the best combination of hyperparameters that minimise MAE.
# This helps reduce overfitting and improves accuracy.

# Define model and parameter grid
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method="hist"
)

param_grid = {
    'n_estimators': [300, 600, 900],
    'learning_rate': [0.02, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0.0, 0.5, 1.0],
    'reg_lambda': [1.0, 2.0]
}

# Use time-series split to preserve order
tscv = TimeSeriesSplit(n_splits=5)

# Use MAE as the scoring metric (lower is better)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

print("Running GridSearchCV — this may take a few minutes...")
grid_search.fit(X_train_s, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest parameters found:\n{grid_search.best_params_}")

# 7. Tests the tuned model on unseen data and reports all three accuracy metrics.

pred_test = best_model.predict(X_test_s)
mae  = mean_absolute_error(y_test, pred_test)
rmse = sqrt(mean_squared_error(y_test, pred_test))
relative_mae = (mae / np.mean(y_test)) * 100

print(f"\nModel Accuracy Summary (Tuned XGBoost):")
print(f" - Mean Absolute Error (MAE): {mae:.3f}")
print(f" - Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f" - Relative MAE: {relative_mae:.2f}% of average realised volatility")
print(f" - Test sample size: {len(y_test)} observations")

# 8. Produces today’s forecast for the next 14 days using the tuned model.

latest_features_s = scaler.transform(df_latest[FEATURES].values)
rv_next14_pred = best_model.predict(latest_features_s)[0]

print(f"\nPredicted NEXT 14-day realised volatility (annualised): {rv_next14_pred:.2f}")
print(f"Latest data date: {df_latest['date'].iloc[0].date()}")

# 9. Saves tuned model and scaler for reuse or deployment.

import joblib
joblib.dump(best_model, "rv_har_xgb_tuned_model.joblib")
joblib.dump(scaler, "rv_feature_scaler.joblib")
df_train.to_csv("rv_training_data.csv", index=False)

#Ways to optimise
#1 Not enough features
#2 Too short a training history
#3 Noise in target variable
