# 📈 Realised Volatility Forecast — HAR + XGBoost Model

This project develops a **machine-learning volatility forecasting engine** that predicts the **next 14-day realised volatility** for the **QQQ ETF (NASDAQ-100)**.  
It fuses **econometric memory structures (HAR model)** with **non-linear ML methods (XGBoost)** and **technical indicators** to generate short-term volatility forecasts for **options trading**, **volatility timing**, and **risk-management** strategies.

---

## 🎯 Objective

Provide a **systematic volatility-forecasting framework** that can:

- Quantify expected **future realised volatility** based on historical patterns and technical factors.  
- Support **volatility-arbitrage**, **gamma-scalping**, and **hedging** strategies.  
- Serve as an analytical module within **options trading dashboards** or **vol-timing bots**.  
- Demonstrate the hybridisation of **HAR regression** and **XGBoost** for volatility prediction.

---

## ⚙️ Methodology

### 1. Data Collection — `rv_forecast_model.py`
Authenticates securely to the **Alpaca Data API** using credentials stored in `.env`, then downloads ~3 years of daily **OHLCV bars** for QQQ.

| Step | Description |
|:--|:--|
| **API Source** | Alpaca Data API (via `alpaca-py`) |
| **Frequency** | Daily |
| **Assets** | QQQ ETF (NASDAQ-100) |
| **Period** | ≈ 3 Years (rolling window) |

---

### 2. Feature Engineering
Transforms the raw data into predictive features capturing market memory and state:

| Feature Type | Description |
|:--|:--|
| **HAR Memory** | 1-, 5- and 22-day realised-volatility lags (`rv_1`, `rv_5`, `rv_22`, `rv_5_22_ratio`) |
| **Momentum & Volatility** | RSI (14), ATR (14), Bollinger Band width |
| **Return & Volume Stats** | 5- and 22-day return stdevs, 20-day volume z-score |
| **Target Variable** | 14-day forward realised volatility (`rv_14_forward`) |

---

### 3. Model Training
Implements a **HAR + XGBoost hybrid**:

- Scales features using `RobustScaler`.  
- Applies **TimeSeriesSplit cross-validation** to preserve chronological order.  
- Runs **GridSearchCV** to tune learning-rate, depth, regularisation, and sampling parameters.  
- Evaluates via **Mean Absolute Error (MAE)** as the primary scoring metric.

---

### 4. Evaluation & Forecasting

| Metric | Description |
|:--|:--|
| **MAE** | Mean Absolute Error between predicted vs actual realised volatility |
| **RMSE** | Root Mean Squared Error – sensitive to outliers |
| **Relative MAE** | Error as a % of average realised volatility |
| **Forecast Output** | Next 14-day annualised realised volatility estimate |

Example output (console):

Predicted NEXT 14-day realised volatility (annualised): 22.45
Latest data date: 2025-11-03

---

## 🧠 Core Technologies

**Python Libraries Used**  
`pandas` • `numpy` • `scikit-learn` • `xgboost` • `ta` • `alpaca-py` • `joblib` • `python-dotenv`

