# =========================================================
# LSTM + XGBoost 14-Day Realised Volatility Forecast — Batch Runner
# Fully Robust | Deterministic | Missing-Data Safe | UTC-Safe | Debug-Aware
# With Time-Decay Weighting & Extended 7-Year Window + VIX/VXN Integration
# + Expanding-Window Walk-Forward Validation (XGB, 14-Day Test Windows)
# + WFA Confidence & Regime-Similarity Confidence (XGB)
# =========================================================

from dotenv import load_dotenv
import os, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import tensorflow as tf

# ML + preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# XGBoost
from xgboost import XGBRegressor

# Indicators
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Data sources
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Option & IV
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq


# ==============================
# CONFIG
# ==============================
DEBUG_MODE = True
SEQ_WINDOW = 22
RV_WINDOW = 14
TRADING_DAYS = 252
TARGET_DAYS_IV = 14

EDGE_VOL_PCTPTS = 2.0
DELTA_VOL_PCTPTS = 1.0

YEARS_HISTORY = 7
LAMBDA_DECAY = 0.4

# ---- Determinism ----
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

load_dotenv()
ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")


# ==============================
# Black–Scholes Functions
# ==============================
def black_scholes_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_iv(S, K, T, r, price):
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if price < intrinsic:
        return np.nan

    def f(sig):
        return black_scholes_call_price(S, K, T, r, sig) - price

    try:
        return brentq(f, 1e-6, 5.0, maxiter=300)
    except:
        return np.nan


# ==============================
# Fetch Latest ATM IV (UTC-Safe)
# ==============================
def get_latest_atm_iv_14d(df_prices, symbol, target_days=TARGET_DAYS_IV, r=0.00):
    ticker = yf.Ticker(symbol)
    try:
        opt_dates = ticker.options
    except:
        return np.nan, None
    if not opt_dates:
        return np.nan, None

    today_utc = datetime.now(timezone.utc)
    expiry = min(
        opt_dates,
        key=lambda d: abs(
            (datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - today_utc).days - target_days
        ),
    )

    try:
        calls = ticker.option_chain(expiry).calls.copy()
    except:
        return np.nan, expiry
    if calls is None or calls.empty:
        return np.nan, expiry

    S = float(df_prices["close"].iloc[-1])
    calls["strike_diff"] = (calls["strike"] - S).abs()

    def best_price(row):
        bid, ask, last = row.get("bid", np.nan), row.get("ask", np.nan), row.get("lastPrice", np.nan)
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        if last > 0:
            return last
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return np.nan

    calls = calls.sort_values("strike_diff")
    T = target_days / 365
    for _, row in calls.iterrows():
        iv = black_scholes_call_iv(S, float(row["strike"]), T, r, best_price(row))
        if np.isfinite(iv) and iv > 0:
            print(f"📅 {symbol}: Nearest expiry {expiry}, IV={iv*100:.2f}%")
            return iv * 100, expiry
    return np.nan, expiry


# ==============================
# Expanding-Window Walk-Forward Validation (XGB, 14-Day Test)
# ==============================
def run_walk_forward_xgb(df_train, FEATURES, row_weights, symbol, out_root, test_window=RV_WINDOW):
    """
    Expanding-window walk-forward validation for XGB:
    - Train on [0 : train_end)
    - Test  on [train_end : train_end + test_window)
    - Step forward by test_window until end of df_train
    Saves: batch_vol_runs/wfa_xgb_<SYMBOL>.csv
    """
    os.makedirs(out_root, exist_ok=True)

    n = len(df_train)
    min_train_size = max(SEQ_WINDOW, 60)

    if n < (min_train_size + test_window):
        print(f"⚠️ Not enough data for XGB walk-forward on {symbol}")
        return

    records = []
    train_end = min_train_size

    while train_end + test_window <= n:
        train_slice = slice(0, train_end)
        test_slice = slice(train_end, train_end + test_window)

        df_tr = df_train.iloc[train_slice]
        df_te = df_train.iloc[test_slice]

        X_tr = df_tr[FEATURES].values
        y_tr = df_tr["rv_14_forward"].values
        X_te = df_te[FEATURES].values
        y_te = df_te["rv_14_forward"].values

        w_tr = row_weights[train_slice]

        xgb_wf = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=SEED,
            tree_method="hist",
            n_jobs=1,
        )
        xgb_wf.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = xgb_wf.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        relmae = (mae / max(np.mean(y_te), 1e-8)) * 100.0

        records.append(
            {
                "Train_End_Date": df_tr["date"].iloc[-1],
                "Test_Start_Date": df_te["date"].iloc[0],
                "Test_End_Date": df_te["date"].iloc[-1],
                "XGB_WFA_MAE": mae,
                "XGB_WFA_RelMAE": relmae,
                "N_Test": len(y_te),
            }
        )

        train_end += test_window

    if records:
        wfa_df = pd.DataFrame(records)
        wfa_path = os.path.join(out_root, f"wfa_xgb_{symbol.upper()}.csv")
        wfa_df.to_csv(wfa_path, index=False)
        print(f"📈 XGB walk-forward results saved → {wfa_path}")


# ==============================
# PIPELINE
# ==============================
def run_pipeline(symbol, out_root="batch_vol_runs"):
    print(f"\n=== Running pipeline for: {symbol} ===")

    client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=YEARS_HISTORY * 365)

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(req).df
    if bars is None or bars.empty:
        raise RuntimeError(f"No data for {symbol}")

    df = (
        bars.xs(symbol, level="symbol")
        .reset_index()
        .rename(columns={"timestamp": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # ---- Merge VIX/VXN ----
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
        vxn = yf.download("^VXN", start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
        if isinstance(vxn.columns, pd.MultiIndex):
            vxn.columns = [c[0] if isinstance(c, tuple) else c for c in vxn.columns]
        close_col_vix = "Close" if "Close" in vix.columns else "Adj Close"
        close_col_vxn = "Close" if "Close" in vxn.columns else "Adj Close"
        vix = (
            vix[[close_col_vix]]
            .rename(columns={close_col_vix: "vix_close"})
            .reset_index()
            .rename(columns={"Date": "date"})
        )
        vxn = (
            vxn[[close_col_vxn]]
            .rename(columns={close_col_vxn: "vxn_close"})
            .reset_index()
            .rename(columns={"Date": "date"})
        )
        vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None)
        vxn["date"] = pd.to_datetime(vxn["date"]).dt.tz_localize(None)
        df = pd.merge(df, vix, on="date", how="left")
        df = pd.merge(df, vxn, on="date", how="left")
        df["vix_vxn_spread"] = df["vix_close"] - df["vxn_close"]
        print(f"✅ Merged VIX/VXN successfully for {symbol}")
    except Exception as e:
        print(f"⚠️ Could not fetch/merge VIX/VXN for {symbol}: {e}")
        df["vix_close"], df["vxn_close"], df["vix_vxn_spread"] = np.nan, np.nan, np.nan

    # ---- IV ----
    iv, _ = get_latest_atm_iv_14d(df, symbol)
    df["IV_14"] = np.nan
    df.loc[df.index[-1], "IV_14"] = iv
    df["IV_14"] = df["IV_14"].bfill().ffill()

    # ---- Feature engineering ----
    df["ret"] = np.log(df["close"]).diff()
    df["RV_14_raw"] = (
        df["ret"]
        .rolling(RV_WINDOW)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
        * np.sqrt(TRADING_DAYS)
    )
    df["RV_14"] = df["RV_14_raw"] * 100
    df["RV_14_percentile"] = df["RV_14"].rank(pct=True) * 100
    df["rv_1"] = df["RV_14_raw"].shift(1)
    df["rv_5"] = (
        df["ret"]
        .rolling(5)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
        * np.sqrt(TRADING_DAYS)
    ).shift(1)
    df["rv_22"] = (
        df["ret"]
        .rolling(22)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
        * np.sqrt(TRADING_DAYS)
    ).shift(1)
    df["rv_63"] = (
        df["ret"]
        .rolling(63)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
        * np.sqrt(TRADING_DAYS)
    ).shift(1)
    df["rv_5_22_ratio"] = df["rv_5"] / df["rv_22"]
    df["RV_14_rollstd_21"] = df["RV_14_raw"].rolling(21).std()
    df["rsi_14"] = RSIIndicator(df["close"], 14).rsi()
    df["rsi_14_deriv"] = df["rsi_14"].diff()
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_22"] = df["close"].pct_change(22)
    df["sma_50"] = df["close"].rolling(50).mean()
    df["trend_strength_50"] = abs(df["close"] - df["sma_50"]) / df["sma_50"]

    def rolling_slope(series, window=20):
        slopes = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            y = np.log(series[i - window : i])
            x = np.arange(window)
            slopes[i] = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
        return slopes

    df["slope_20"] = rolling_slope(df["close"], 20)
    df["vol_z_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    df["volume_ratio_5_20"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    df["turnover"] = df["close"] * df["volume"]
    df["liquidity_pressure"] = (df["high"] - df["low"]) / df["volume"]
    bb = BollingerBands(df["close"], 20, 2)
    df["bb_bandwidth"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["bb_pctB"] = (df["close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    df["atr_pct"] = df["atr_14"] / df["close"] * 100
    df["IV_14_lag1"] = df["IV_14"].shift(1)
    df["IV_RV_spread_lag1"] = df["IV_14_lag1"] - df["RV_14"].shift(1)
    df["IV_trend_5"] = df["IV_14"].rolling(5).apply(
        lambda x: np.polyfit(range(len(x.dropna())), x.dropna(), 1)[0] if len(x.dropna()) > 1 else np.nan
    )
    df["IV_volatility_10"] = df["IV_14"].rolling(10).std()
    df["rsi_bb_interact"] = df["rsi_14"] * df["bb_bandwidth"]
    df["rv_vol_interact"] = df["rv_5_22_ratio"] * df["vol_z_20"]
    df["ret_std_5"] = df["ret"].rolling(5).std() * np.sqrt(TRADING_DAYS)
    df["ret_std_22"] = df["ret"].rolling(22).std() * np.sqrt(TRADING_DAYS)
    df["rv_14_forward"] = df["RV_14_raw"].shift(-RV_WINDOW)
    df = df.iloc[100:].copy()

    FEATURES = [
        "rv_1",
        "rv_5",
        "rv_22",
        "rv_63",
        "rv_5_22_ratio",
        "RV_14_rollstd_21",
        "rsi_14",
        "rsi_14_deriv",
        "momentum_10",
        "momentum_22",
        "trend_strength_50",
        "slope_20",
        "vol_z_20",
        "volume_ratio_5_20",
        "turnover",
        "liquidity_pressure",
        "bb_bandwidth",
        "bb_pctB",
        "atr_14",
        "atr_pct",
        "IV_14_lag1",
        "IV_RV_spread_lag1",
        "IV_trend_5",
        "IV_volatility_10",
        "rsi_bb_interact",
        "rv_vol_interact",
        "ret_std_5",
        "ret_std_22",
        "vix_close",
        "vxn_close",
        "vix_vxn_spread",
    ]

    for col in FEATURES.copy():
        if col not in df.columns or df[col].isna().all():
            FEATURES.remove(col)

    # ---- SAVE FEATURE SET ----
    os.makedirs(out_root, exist_ok=True)
    feature_df = df["date"].to_frame().join(df[FEATURES])
    feature_out_path = os.path.join(out_root, f"feature_set_{symbol.upper()}.csv")
    feature_df.to_csv(feature_out_path, index=False)
    print(f"🧩 Saved feature dataset → {feature_out_path} ({len(FEATURES)} features)")

    df_train = df.dropna(subset=["rv_14_forward"]).copy()
    if df_train.empty:
        raise RuntimeError(f"Not enough data after feature engineering for {symbol}")
    df_train[FEATURES] = df_train[FEATURES].fillna(df_train[FEATURES].median())

    df_latest = df.dropna(subset=FEATURES)
    if df_latest.empty:
        raise RuntimeError(f"No valid rows for forecasting for {symbol}")

    # ---- DEBUG ----
    if DEBUG_MODE:
        na_counts = df[FEATURES].isna().sum()
        na_pct = (na_counts / len(df)) * 100
        summary = (
            pd.DataFrame({"NaNs": na_counts, "%_missing": na_pct})
            .sort_values("NaNs", ascending=False)
        )
        summary.to_csv(os.path.join(out_root, f"feature_health_{symbol.upper()}.csv"))

    # ---- Time-decay weights ----
    age_days = (df_train["date"].max() - df_train["date"]).dt.days
    age_years = age_days / 365.0
    row_weights = np.exp(-LAMBDA_DECAY * age_years).values

    # ---- Walk-Forward Validation (XGB, expanding window, 14-day test) ----
    run_walk_forward_xgb(df_train, FEATURES, row_weights, symbol, out_root)

    # ---- WFA Confidence Score (XGB, recent vs historical) ----
    wfa_confidence = np.nan
    wfa_path = os.path.join(out_root, f"wfa_xgb_{symbol.upper()}.csv")
    if os.path.exists(wfa_path):
        try:
            wfa_df_conf = pd.read_csv(wfa_path)
            if "XGB_WFA_RelMAE" in wfa_df_conf.columns:
                rel_errors = wfa_df_conf["XGB_WFA_RelMAE"].dropna().values
                if rel_errors.size > 0:
                    N = 10
                    recent = rel_errors[-N:] if rel_errors.size >= N else rel_errors
                    recent_mean = recent.mean()
                    low = np.percentile(rel_errors, 10)
                    high = np.percentile(rel_errors, 90)
                    if high > low:
                        norm = (recent_mean - low) / (high - low)
                        norm = max(0.0, min(1.0, norm))
                        wfa_confidence = (1.0 - norm) * 100.0
        except Exception as e:
            print(f"⚠️ Could not compute WFA confidence for {symbol}: {e}")

    # ---- Sequences ----
    X = df_train[FEATURES].values
    y = df_train["rv_14_forward"].values
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)

    def to_sequences(X_arr, y_arr, win=SEQ_WINDOW):
        Xs, ys = [], []
        for i in range(win, len(X_arr)):
            Xs.append(X_arr[i - win : i])
            ys.append(y_arr[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = to_sequences(X_s, y, SEQ_WINDOW)
    seq_weights = row_weights[SEQ_WINDOW:]
    cut = int(len(X_seq) * 0.8)
    X_train, y_train, X_test, y_test = (
        X_seq[:cut],
        y_seq[:cut],
        X_seq[cut:],
        y_seq[cut:],
    )
    w_train = seq_weights[: len(y_train)]

    # ---- LSTM ----
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(SEQ_WINDOW, X_train.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mae")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        sample_weight=w_train,
        epochs=60,
        batch_size=16,
        callbacks=[es],
        verbose=0,
    )

    y_pred_lstm = model.predict(X_test, verbose=0).flatten()
    lstm_mae = mean_absolute_error(y_test, y_pred_lstm)
    lstm_relmae = (lstm_mae / max(np.mean(y_test), 1e-8)) * 100.0

    latest_block = scaler.transform(df_latest[FEATURES].values)[-SEQ_WINDOW:].reshape(
        1, SEQ_WINDOW, len(FEATURES)
    )
    lstm_forecast_raw = model.predict(latest_block, verbose=0)[0][0]
    lstm_forecast_pct = lstm_forecast_raw * 100.0

    # ---- XGB ----
    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
    )
    cut_tab = int(len(df_train) * 0.8)
    X_train_tab = df_train.iloc[:cut_tab][FEATURES].values
    y_train_tab = df_train.iloc[:cut_tab]["rv_14_forward"].values
    X_test_tab = df_train.iloc[cut_tab:][FEATURES].values
    y_test_tab = df_train.iloc[cut_tab:]["rv_14_forward"].values
    w_tab = row_weights[: len(y_train_tab)]

    xgb.fit(X_train_tab, y_train_tab, sample_weight=w_tab)
    y_pred_xgb = xgb.predict(X_test_tab)
    xgb_mae = mean_absolute_error(y_test_tab, y_pred_xgb)
    xgb_relmae = (xgb_mae / max(np.mean(y_test_tab), 1e-8)) * 100.0
    xgb_forecast_pct = float(
        xgb.predict(df_latest[FEATURES].values[-1].reshape(1, -1))[0]
    ) * 100.0

    # ---- Regime-Similarity Confidence (XGB) ----
    regime_confidence = np.nan
    if os.path.exists(wfa_path):
        try:
            wfa_df_reg = pd.read_csv(wfa_path)
            if (
                "XGB_WFA_RelMAE" in wfa_df_reg.columns
                and "Test_Start_Date" in wfa_df_reg.columns
            ):
                wfa_df_reg["Test_Start_Date"] = pd.to_datetime(
                    wfa_df_reg["Test_Start_Date"]
                )
                df_train_for_merge = df_train[["date"] + FEATURES].copy()
                merged = pd.merge(
                    wfa_df_reg,
                    df_train_for_merge,
                    left_on="Test_Start_Date",
                    right_on="date",
                    how="inner",
                )
                merged = merged.dropna(subset=["XGB_WFA_RelMAE"])
                if not merged.empty:
                    # scale historical regime features
                    hist_feats = merged[FEATURES].values
                    hist_feats_scaled = scaler.transform(hist_feats)

                    # current regime (use same latest row as XGB forecast)
                    current_feats = df_latest[FEATURES].values[-1].reshape(1, -1)
                    current_feats_scaled = scaler.transform(current_feats)[0]

                    # distances in feature space
                    diffs_reg = hist_feats_scaled - current_feats_scaled
                    dists = np.linalg.norm(diffs_reg, axis=1)

                    K = min(10, len(merged))
                    idx = np.argsort(dists)[:K]
                    similar_errors = merged.iloc[idx]["XGB_WFA_RelMAE"].values

                    if similar_errors.size > 0:
                        similar_mean = similar_errors.mean()
                        all_errors = merged["XGB_WFA_RelMAE"].values
                        low_r = np.percentile(all_errors, 10)
                        high_r = np.percentile(all_errors, 90)
                        if high_r > low_r:
                            norm_r = (similar_mean - low_r) / (high_r - low_r)
                            norm_r = max(0.0, min(1.0, norm_r))
                            regime_confidence = (1.0 - norm_r) * 100.0
        except Exception as e:
            print(f"⚠️ Could not compute regime similarity confidence for {symbol}: {e}")

    df["XGB_Regime_Confidence"] = np.nan
    df.loc[df.index[-1], "XGB_Regime_Confidence"] = regime_confidence

    # ---- Mahalanobis Drift Score (meta-metric, not a feature) ----
    mahalanobis_drift_score = np.nan
    try:
        feat_matrix = df_train[FEATURES].copy()
        if len(feat_matrix) > len(FEATURES):
            mu = feat_matrix.mean(axis=0)
            cov = np.cov(feat_matrix.values, rowvar=False)
            eps = 1e-6
            cov_reg = cov + eps * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov_reg)

            df_all_feats = df[FEATURES].copy().fillna(mu)
            diffs_all = df_all_feats.values - mu.values
            d_mahal = np.sqrt(np.einsum("ij,jk,ik->i", diffs_all, inv_cov, diffs_all))

            med = np.nanmedian(d_mahal)
            if med == 0 or np.isnan(med):
                df["Mahalanobis_Drift_Score"] = np.nan
            else:
                df["Mahalanobis_Drift_Score"] = d_mahal / med

            mahalanobis_drift_score = float(df["Mahalanobis_Drift_Score"].iloc[-1])
        else:
            df["Mahalanobis_Drift_Score"] = np.nan
    except Exception as e:
        print(f"⚠️ Could not compute Mahalanobis Drift Score for {symbol}: {e}")
        df["Mahalanobis_Drift_Score"] = np.nan

    # ---- Ensemble (Regime-Adaptive via WFA + Regime Similarity + Drift) ----
    lstm_w_error = 1.0 / max(lstm_mae, 1e-8)
    xgb_w_error = 1.0 / max(xgb_mae, 1e-8)

    # WFA-based component (0–100 → 0–1)
    if (wfa_confidence is not None) and (not np.isnan(wfa_confidence)):
        wfa_norm = np.clip(wfa_confidence / 100.0, 0.0, 1.0)
    else:
        wfa_norm = 0.5  # neutral if unknown

    lstm_wfa_score = max(1.0 - wfa_norm, 0.1)
    xgb_wfa_score = max(wfa_norm, 0.1)

    # Regime-Similarity component (0–100 → 0–1)
    if (regime_confidence is not None) and (not np.isnan(regime_confidence)):
        reg_norm = np.clip(regime_confidence / 100.0, 0.0, 1.0)
    else:
        reg_norm = 0.5

    lstm_regime_score = max(1.0 - reg_norm, 0.1)
    xgb_regime_score = max(reg_norm, 0.1)

    # Drift reliability component (symmetric penalty, 1.0 good → 0.0 bad)
    if (mahalanobis_drift_score is not None) and (not np.isnan(mahalanobis_drift_score)):
        drift = mahalanobis_drift_score
        if drift <= 1.0:
            drift_rel_01 = 1.0
        elif drift >= 4.0:
            drift_rel_01 = 0.0
        else:
            drift_rel_01 = (4.0 - drift) / (4.0 - 1.0)
        drift_factor = max(drift_rel_01, 0.1)
    else:
        drift_factor = 1.0

    # Final raw weights (multiplicative E1: equal strength for each signal)
    w_lstm_final = (
        lstm_w_error * lstm_wfa_score * lstm_regime_score * drift_factor
    )
    w_xgb_final = (
        xgb_w_error * xgb_wfa_score * xgb_regime_score * drift_factor
    )
    denom_w = w_lstm_final + w_xgb_final

    if denom_w > 0:
        lstm_weight = w_lstm_final / denom_w
        xgb_weight = w_xgb_final / denom_w
    else:
        # Fallback to simple inverse-MAE if something goes wrong
        lstm_w = lstm_w_error
        xgb_w = xgb_w_error
        lstm_weight = lstm_w / (lstm_w + xgb_w)
        xgb_weight = xgb_w / (lstm_w + xgb_w)

    ens_forecast_pct = lstm_weight * lstm_forecast_pct + xgb_weight * xgb_forecast_pct

    # ---- Outputs ----
    df.loc[df.index[-1], [
        "Forecast_RV_14",
        "XGB_Forecast_RV_14",
        "Ensemble_Forecast_RV_14",
        "LSTM_Weight",
        "XGB_Weight",
        "Relative_MAE_LSTM",
        "Relative_MAE_XGB",
    ]] = [
        lstm_forecast_pct,
        xgb_forecast_pct,
        ens_forecast_pct,
        lstm_weight,
        xgb_weight,
        lstm_relmae,
        xgb_relmae,
    ]

    df["IV_minus_RV"] = df["IV_14"] - df["RV_14"]
    df["Forecast_vs_IV"] = df["IV_14"] - df["Forecast_RV_14"]
    df["XGB_vs_IV"] = df["IV_14"] - df["XGB_Forecast_RV_14"]
    df["Ensemble_vs_IV"] = df["IV_14"] - df["Ensemble_Forecast_RV_14"]

    df["Model_Disagreement"] = abs(lstm_forecast_pct - xgb_forecast_pct)
    current_rv_pct = float(df["RV_14"].iloc[-1])
    df["LSTM_Delta_RV"] = lstm_forecast_pct - current_rv_pct
    df["XGB_Delta_RV"] = xgb_forecast_pct - current_rv_pct

    # ---- Percentiles for compression metrics ----
    df["bb_bandwidth_pctile"] = df["bb_bandwidth"].rank(pct=True) * 100
    df["atr_pct_pctile"] = df["atr_pct"].rank(pct=True) * 100

    df["ShortVega_Signal"] = int(df["Ensemble_vs_IV"].iloc[-1] >= EDGE_VOL_PCTPTS)

    df["End_LongGamma_Flag"] = int(
        (df["LSTM_Delta_RV"].iloc[-1] < -DELTA_VOL_PCTPTS)
        and (df["XGB_Delta_RV"].iloc[-1] < -DELTA_VOL_PCTPTS)
    )

    # === SURGICAL CHANGE BELOW ===
    df["Enter_LongGamma_Signal"] = int(
        (df["Ensemble_vs_IV"].iloc[-1] <= -EDGE_VOL_PCTPTS)
        and (df["LSTM_Delta_RV"].iloc[-1] >= DELTA_VOL_PCTPTS)
        and (df["XGB_Delta_RV"].iloc[-1] >= DELTA_VOL_PCTPTS)
        and (df["bb_bandwidth_pctile"].iloc[-1] <= 20.0)
        and (df["atr_pct_pctile"].iloc[-1] <= 30.0)
    )
    # ==============================

    # attach WFA confidence to the last row
    df["XGB_WFA_Confidence"] = np.nan
    df.loc[df.index[-1], "XGB_WFA_Confidence"] = wfa_confidence

    # ---- Model Reliability Score (MRS) combining WFA, Regime & Drift ----
    mrs_score = np.nan
    df["Model_Reliability_Score"] = np.nan
    try:
        if (
            (wfa_confidence is not None)
            and (not np.isnan(wfa_confidence))
            and (regime_confidence is not None)
            and (not np.isnan(regime_confidence))
            and (mahalanobis_drift_score is not None)
            and (not np.isnan(mahalanobis_drift_score))
        ):
            # Drift reliability: 100 at drift<=1, 0 at drift>=4, linear in-between
            drift = mahalanobis_drift_score
            if drift <= 1.0:
                drift_rel = 100.0
            elif drift >= 4.0:
                drift_rel = 0.0
            else:
                drift_rel = 100.0 * (4.0 - drift) / (4.0 - 1.0)

            mrs = (
                0.4 * float(wfa_confidence)
                + 0.4 * float(regime_confidence)
                + 0.2 * float(drift_rel)
            )
            mrs = max(0.0, min(100.0, mrs))
            mrs_score = mrs
            df.loc[df.index[-1], "Model_Reliability_Score"] = mrs_score
    except Exception as e:
        print(f"⚠️ Could not compute Model Reliability Score for {symbol}: {e}")

    dashboard_cols = [
        "date",
        "close",
        "RV_14",
        "RV_14_percentile",
        "IV_14",
        "IV_minus_RV",
        "Forecast_RV_14",
        "Forecast_vs_IV",
        "XGB_Forecast_RV_14",
        "XGB_vs_IV",
        "Ensemble_Forecast_RV_14",
        "Ensemble_vs_IV",
        "LSTM_Weight",
        "XGB_Weight",
        "Relative_MAE_LSTM",
        "Relative_MAE_XGB",
        "Model_Disagreement",
        "LSTM_Delta_RV",
        "XGB_Delta_RV",
        "bb_bandwidth",
        "bb_bandwidth_pctile",
        "atr_14",
        "atr_pct",
        "atr_pct_pctile",
        "vol_z_20",
        "ret_std_5",
        "ret_std_22",
        "ShortVega_Signal",
        "End_LongGamma_Flag",
        "Enter_LongGamma_Signal",
        "XGB_WFA_Confidence",
        "XGB_Regime_Confidence",
        "Mahalanobis_Drift_Score",
        "Model_Reliability_Score",
    ]

    out = df[dashboard_cols].dropna(subset=["RV_14"])
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"vol_dashboard_data_{symbol.upper()}.csv")
    out.to_csv(out_path, index=False)
    print(f"✅ Saved → {out_path}")

    latest = out.iloc[-1]
    return {
        "symbol": symbol.upper(),
        "IV_14": latest["IV_14"],
        "Forecast_RV_14": latest["Forecast_RV_14"],
        "XGB_Forecast_RV_14": latest["XGB_Forecast_RV_14"],
        "Ensemble_Forecast_RV_14": latest["Ensemble_Forecast_RV_14"],
        "LSTM_Weight": latest["LSTM_Weight"],
        "XGB_Weight": latest["XGB_Weight"],
        "Relative_MAE_LSTM": latest["Relative_MAE_LSTM"],
        "Relative_MAE_XGB": latest["Relative_MAE_XGB"],
        "Ensemble_vs_IV": latest["Ensemble_vs_IV"],
        "ShortVega_Signal": latest["ShortVega_Signal"],
        "End_LongGamma_Flag": latest["End_LongGamma_Flag"],
        "Enter_LongGamma_Signal": latest["Enter_LongGamma_Signal"],
        "XGB_WFA_Confidence": latest["XGB_WFA_Confidence"],
        "XGB_Regime_Confidence": latest["XGB_Regime_Confidence"],
        "Mahalanobis_Drift_Score": latest["Mahalanobis_Drift_Score"],
        "Model_Reliability_Score": latest["Model_Reliability_Score"],
    }


# ==============================
# Batch Runner
# ==============================
if __name__ == "__main__":
    SYMBOLS_CSV = "symbols.csv"
    OUT_ROOT = "batch_vol_runs"

    tickers = (
        pd.read_csv(SYMBOLS_CSV)
        .iloc[:, 0]
        .astype(str)
        .str.strip()
        .dropna()
        .tolist()
    )
    summary = []

    for sym in tickers:
        try:
            rec = run_pipeline(sym, OUT_ROOT)
            summary.append(rec)
        except Exception as e:
            print(f"❌ {sym}: {e}")

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(OUT_ROOT, "summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n📊 Summary metrics saved → {summary_path}\n")
