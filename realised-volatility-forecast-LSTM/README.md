# 📈 Realised Volatility Forecast — HAR + LSTM Sequence Model

This project develops a **time-dependent volatility forecasting engine** that predicts the **next 14-day realised volatility** for the **QQQ ETF (NASDAQ-100)**.  
It enhances the HAR (Heterogeneous Autoregressive) volatility memory structure with a **22-day LSTM recurrent neural network**, enabling the model to learn **volatility clustering and market regime dynamics** directly from historical behaviour.

This approach is applicable to **volatility-arbitrage**, **gamma-scalping**, **hedging**, and **options spread optimisation** workflows.

---

## 🎯 Objective

Develop a **systematic volatility forecasting framework** that:

- Models **future volatility** using both **market state** and **temporal market structure**.
- Identifies **volatility compression**, **expansion**, and **regime shift** phases.
- Provides input signals for **options trade structuring and vol-timing systems**.
- Demonstrates the upgrade pathway from **static HAR/XGBoost models → sequence-aware LSTM models**.

---

## 🌐 System Architecture

QQQ Data (Alpaca API) → Feature Engineering → 22-Day Rolling Window → LSTM Model → Forward 14-Day RV Forecast

## ⚙️ Methodology

### 1. Data Collection — `rv_har_lstm_seq_model.py`

| Step | Description |
|:--|:--|
| **API Source** | Alpaca Data API (`alpaca-py`) |
| **Frequency** | Daily bars |
| **Asset** | QQQ (NASDAQ-100 ETF) |
| **Lookback Horizon** | ~3 years |

The model automatically fetches, cleans, and structures price data into a modelling-ready DataFrame.

---

### 2. Feature Engineering

| Feature Type | Description |
|:--|:--|
| **HAR Memory** | 1-day, 5-day, and 22-day realised volatility lags + ratio (`rv_1`, `rv_5`, `rv_22`, `rv_5_22_ratio`) |
| **Momentum + Volatility** | RSI(14), ATR(14), Bollinger Bandwidth |
| **Return Variability** | Rolling σ of log returns (5- and 22-day) |
| **Market Participation** | 20-day volume Z-score |
| **Target** | 14-day forward realised volatility (`rv_14_forward`) |

![Feature Chart](/images/feature_visualisation.png)

---

### 3. Sequence Construction (Key Improvement)

Instead of modelling **single-day features**, the network consumes **22-day rolling windows**:

Input Tensor Shape: (22 time steps × feature dimensions)
Output Target: Forward 14-day realised volatility estimate

---

### 4. Model Training — LSTM Sequence Model

| Layer | Role |
|:--|:--|
| `LSTM(64)` | Learns temporal patterns in volatility regimes |
| `LSTM(32)` | Refines learned structure |
| `Dropout(0.2)` | Regularisation against regime-fit overtraining |
| `Dense(16)` | Feature compression |
| `Dense(1)` | Final volatility forecast output |

Loss function: **MAE** (stable under heavy-tailed financial noise)

---

## 📊 Evaluation & Forecasting

| Metric | Meaning |
|:--|:--|
| **MAE** | Average forecast error |
| **RMSE** | Penalises larger deviation events |
| **Relative MAE** | Forecast error relative to average RV level |
| **Forecast Output** | Annualised 14-day forward realised volatility |

Example output:

Predicted NEXT 14-day realised volatility (annualised): 19.87
Latest data date: 2025-11-03

---

## 🧠 Core Technologies

`python` • `pandas` • `numpy` • `scikit-learn` • `tensorflow.keras` • `ta` • `alpaca-py` • `joblib` • `dotenv`

---
