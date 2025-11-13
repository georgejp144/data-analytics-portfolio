📈 14-Day Realised Volatility Forecasting Engine
LSTM + XGBoost + Regime Awareness + IV Integration + Ensemble Model

This project implements a state-of-the-art volatility forecasting system designed to predict 14-day realised volatility (RV) across equity tickers (NASDAQ-100 and beyond).
It extends classic HAR-style modelling with:

LSTM sequence learning

XGBoost gradient boosting

Expanding-window walk-forward validation

Regime-similarity scoring

Mahalanobis drift detection

ATM 14-DTE implied volatility extraction

Adaptive ensemble weighting

Volatility dislocation & trade signal generation

The framework is suitable for volatility arbitrage, gamma scalping, long-gamma timing, short-vega strategies, risk forecasting, and options structure optimisation.

🎯 Objective

Build a full volatility forecasting pipeline that:

Predicts future realised volatility with regime-aware model confidence.

Detects IV–RV dislocations for long-gamma / short-vega opportunities.

Incorporates implied volatility (IV) via a custom Black–Scholes inversion engine.

Measures forecast reliability using walk-forward validation + drift scoring.

Generates dashboard-ready datasets for trading desks and analytics tools.

🌐 System Architecture

Ticker Data (Alpaca API)
→ VIX/VXN Merge
→ ATM 14-DTE IV Extraction
→ Feature Engineering (60+ signals)
→ Train/Test Split
→ LSTM Sequence Model (22-day windows)
→ XGBoost Tabular Model
→ Walk-Forward Validation
→ Regime Similarity Search
→ Drift Scoring
→ Adaptive Ensemble Forecast
→ Volatility Dislocation Metrics
→ Long/Short Vol Signals
→ Dashboard Output (CSV)

⚙️ Methodology
1. Data Collection (7-Year Window)
Component	Description
API Source	Alpaca StockHistoricalDataClient
Supplementary Data	VIX + VXN (Yahoo Finance)
Timeframe	Daily bars
Symbols	Loaded from symbols.csv
IV Source	Yahoo Finance option chains + Black–Scholes inversion

The pipeline is UTC-safe, missing-data tolerant, and fully deterministic.

2. Implied Volatility Engine (ATM 14-DTE)

The model builds a daily ATM implied volatility series by:

Fetching nearest-expiry option chains

Reconstructing mid-prices (bid/ask/last logic)

Running Black–Scholes call IV inversion

Selecting strikes nearest to spot

Fallback proxy using VIX/VXN scaling

Forward/backward fill for continuity

Output signal: IV_14 (%)

3. Feature Engineering (60+ Signals)
Category	Examples
RV / HAR Components	rv_1, rv_5, rv_22, rv_63, rv_5_22_ratio
Volatility Structure	RV_14, RV_14_percentile, RV_14_rollstd_21
Market Momentum	momentum_10, momentum_22
Volatility Indicators	ATR(14), ATR %, Bollinger Bandwidth, %B
RSI Structure	rsi_14, rsi_14_deriv
Microstructure	vol_z_20, volume_ratio_5_20, turnover, liquidity_pressure
Return Dispersion	ret_std_5, ret_std_22
IV Metrics	IV_14_lag1, IV–RV spread lag, IV trend, IV volatility
Interactions	rsi_bb_interact, rv_vol_interact
Macro Vol Inputs	vix_close, vxn_close, vix_vxn_spread

Target variable: rv_14_forward (raw RV shifted −14 days)

🔁 Sequence Construction (LSTM Input)

The LSTM consumes 22-day rolling windows of all features.

Input Tensor:
(22 time steps × number of engineered features)

Target:
Forward 14-day realised volatility (annualised)

This lets the model learn volatility clustering, momentum shocks, compression regimes, and vol-cycle transitions.

🤖 Model Training
1. LSTM Sequence Model
Layer	Purpose
LSTM(64, return_sequences=True)	Learns temporal structure
Dropout(0.2)	Prevents regime overfitting
LSTM(32)	Extracts mid-term volatility dynamics
Dense(16, relu)	Feature compression
Dense(1)	Final 14-day RV forecast

Loss: MAE (robust to heavy tails)

Validation split: 80/20

Early stopping enabled

Time-decay weighted samples

2. XGBoost Tabular Model

XGBRegressor with:

400 trees

depth=4

hist tree method

learning_rate=0.05

Weighted by time-decay

Tabular feature set identical to LSTM

Outputs:
XGB_Forecast_RV_14, validation MAE, Relative MAE

📊 Walk-Forward Validation (Expanding Window)

Each model undergoes time-consistent testing:

Train on [start → T]

Test on next 14-day forward window

Slide T forward

Repeat until end

Metrics saved as:
wfa_xgb_<SYMBOL>.csv

Outputs:

XGB_WFA_MAE

XGB_WFA_RelMAE

Historical error distribution

🔎 Model Confidence Layers
1. WFA Confidence

Uses historical error distribution to measure:

“How well is XGB performing recently relative to history?”

Produces 0 → 100 score.

2. Regime Similarity Confidence

Current feature vector is compared to historical states.
Nearest 10 regimes extracted → average historical error calculated.

Measures:

“Does this feature regime resemble periods where the model performed well?”

Also 0 → 100 score.

3. Mahalanobis Drift Score

Multivariate drift detection:

Low drift = stable regime

High drift = out-of-distribution warning

Used as a penalty factor for the ensemble.

🧮 Ensemble Forecasting (Regime-Adaptive)

LSTM and XGB forecasts are blended using:

Inverse-MAE weighting

WFA confidence

Regime similarity confidence

Drift reliability factor

Final output:

Ensemble_Forecast_RV_14

Also:

LSTM_Weight

XGB_Weight

Model_Reliability_Score (0 → 100)

⚡ Volatility Dislocation Metrics

Generated for each symbol:

Metric	Meaning
IV_minus_RV	Current IV minus current RV
Forecast_vs_IV	IV relative to LSTM forecast
XGB_vs_IV	IV relative to XGB forecast
Ensemble_vs_IV	IV relative to blended forecast
Model_Disagreement	Forecast spread (LSTM vs XGB)

These quantify edge, compression, mean-reversion pressure, and risk skew.

🎯 Trading Signals

The system generates three primary strategy flags:

1. Short Vega Signal

Triggered when IV is significantly above forecast.

2. End Long Gamma Flag

Triggered when both models predict declining RV.

3. Enter Long Gamma Signal (High-Quality Setup)

Requires ALL:

IV deep below forecast (negative edge threshold)

LSTM_Delta_RV > threshold

XGB_Delta_RV > threshold

Bollinger Bandwidth percentile ≤ 20

ATR percentile ≤ 30

This models volatility compression + expected RV expansion, a classic setup for long-gamma scalping / straddles.

🗂️ Output Files (per symbol)
File	Description
feature_set_<TICKER>.csv	Full engineered feature matrix
feature_health_<TICKER>.csv	Missingness report
wfa_xgb_<TICKER>.csv	Walk-forward validation results
vol_dashboard_data_<TICKER>.csv	Final dashboard-ready dataset
summary_metrics.csv	Batch summary across all tickers
🧠 Core Technologies

python • pandas • numpy • scikit-learn
tensorflow.keras • XGBoost • ta • scipy
alpaca-py • yfinance • dotenv • RobustScaler

✔️ Applications

Volatility arbitrage research

Long-gamma systematic timing

Vega strategies

Options pricing & hedging models

Macro-vol dashboards

Machine-learning volatility studies

Regime detection research
