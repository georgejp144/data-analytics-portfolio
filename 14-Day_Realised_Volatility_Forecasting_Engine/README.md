📈 14-Day Realised Volatility Forecasting System
Hybrid LSTM + XGBoost | Walk-Forward Validation | Regime Similarity | Feature Drift | IV-RV Vol Arbitrage Signals

This repository contains a production-grade volatility forecasting engine used to estimate the next 14-day realised volatility for equity tickers (QQQ, SPY, AAPL, NVDA, etc.).
It uses a hybrid ML architecture that blends:

LSTM sequence model – captures long-memory behaviour in volatility

XGBoost regression model – captures non-linear interactions

Regime-Adaptive weighting – confidence-weighted ensemble

Walk-Forward Validation (Expanding Window) – realistic, non-leaky evaluation

Regime Similarity Score – quantifies how similar today is to past regimes

Feature Drift Score – detects when features are out-of-distribution

IV Proxy Reconstruction – for full 7-year historical consistency

Outputs are stored automatically as CSVs in /batch_vol_runs.

🚀 Why This Project Exists

Forecasting future realised volatility is crucial for:

Gamma scalping / delta-neutral options strategies

IV-RV mean-reversion trades

Risk management & portfolio hedging

Identifying volatility compression & expansion cycles

Traditional models (GARCH, HAR) struggle in modern markets.
This system aims to deliver stable, regime-aware, ML-driven forecasts suitable for real trading workflows.

🧠 Core Features
📌 1. 7-Year Historical Dataset (Deterministic, Missing-Data Safe)

Everything is reproducible with fixed seeds, UTC-safe timestamps, and automatic data cleaning.

📌 2. Full Feature Engineering Suite

Including:

Returns (1d, 3d, 5d, 21d)

Realised volatility (1–21d windows)

VIX & VXN integration

RSI

Bollinger Band features

ATR

HV/IV ratios

Vol compression metrics

Macro event placeholders

Implied Volatility Proxy reconstruction

📌 3. Hybrid Forecasting Architecture

LSTM for volatility clustering

XGB for tree-based pattern learning

Ensemble using confidence weights

📌 4. Walk-Forward Validation

Expanding window logic with 14-day test windows — matching the target horizon.

Provides:

Rolling MAE

Regime-average errors

True out-of-sample tests

📌 5. Regime Similarity Engine

Measures the cosine similarity between current features and historical states.

Returns a 0–1 score indicating how “familiar” the regime is.

High similarity = model more trustworthy.
Low similarity = regime unfamiliar → possible macro shift.

📌 6. Feature Drift Score

Tracks if today’s feature vector sits outside the training distribution using:

Normalised means

Covariance distance

Mahalanobis scoring

If drift > threshold → model warns that forecast reliability is reduced.

📌 7. Full Feature Dataset Export

For each batch run, the system exports:

batch_vol_runs/FEATURES_<TICKER>.csv


This contains every engineered feature for full transparency.

📌 8. Daily Forecast Output

Each ticker outputs:

Date
RV_14
Pred_LSTM
Pred_XGB
Ensemble_Forecast
WFA_Confidence
Regime_Similarity
Feature_Drift
IV_Proxy
IV_RV_Spread

🏗️ System Architecture
                ┌────────────────────────┐
                │  Alpaca / Yahoo Data   │
                └─────────────┬──────────┘
                              │
                       Raw OHLCV Bars
                              │
                              ▼
                ┌────────────────────────┐
                │   Feature Engineering   │
                │  + IV Proxy Rebuild     │
                └─────────────┬──────────┘
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
   ┌────────────────────┐                ┌────────────────────┐
   │       LSTM         │                │       XGBoost       │
   └───────────┬────────┘                └──────────┬─────────┘
               │                                      │
               └──────────────┬───────────────────────┘
                              ▼
                   ┌───────────────────┐
                   │   WFA Engine      │
                   │ + Regime Similarity│
                   │ + Feature Drift    │
                   └─────────┬─────────┘
                              ▼
                 ┌────────────────────────┐
                 │   Ensemble Forecaster   │
                 └─────────────┬──────────┘
                              ▼
                   Output CSVs & Feature Sets


🧪 Validation & Performance
✔️ Expanding-window walk-forward
✔️ Rolling 14-day out-of-sample
✔️ Error tracking by regime cluster
✔️ Residual diagnostics
✔️ LSTM vs XGB divergence checks
✔️ Drift thresholds to flag unstable periods

This mimics real quant research workflow.
