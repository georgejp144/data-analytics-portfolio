# 📈 Options Swing-Trading Indicator — XGBoost & LSTM + Volatility Forecast Integration

This project develops a **machine-learning signal engine** that blends price-based classification (**XGBoost** & **LSTM**) with **realised vs implied volatility forecasts** to guide options swing-trading decisions.  
The system converts multi-model outputs into actionable trade signals such as **bull call**, **bear put**, or **neutral calendar** spreads.

---

## 🎯 Objective

Forecast short-term **directional profitability** and **volatility conditions** for large-cap NASDAQ stocks, then map those combined forecasts to the most efficient **option spread structure**.

---

## ⚙️ Methodology

### 1. Data Collection
- Daily OHLCV data via **yfinance** for major NASDAQ equities (AAPL, MSFT, NVDA, QQQ etc.).
- Realised volatility aggregated from intraday returns.
- Implied volatility sourced from option chains or **VIX** proxies.

### 2. Feature Engineering
- 50+ technical indicators (RSI, MACD, ATR, Bollinger Bands, MFI, etc.)
- Lagged realised and implied volatility terms.
- Market sentiment & breadth ratios where available.

### 3. Predictive Models

| Model | Purpose | Key Inputs |
|:--|:--|:--|
| **XGBoost Classifier** | Captures non-linear relationships between TA features and 3-day returns | RSI, MACD, ATR, ROC, Price–MA ratios |
| **LSTM Network** | Learns temporal dependencies in sequential price data | Normalised OHLC series & volatility lags |
| **Volatility Forecast (HAR / XGBoost)** | Estimates next-period realised volatility | Realised Vol(1, 5, 22), Implied Vol, Volume |

### 4. Signal Fusion
- Combine **XGBoost** and **LSTM** probabilities using weighted averaging.  
- Overlay volatility forecast to classify regime: **Low Vol**, **Rising Vol**, **High Vol**.

### 5. Strategy Mapping

| Direction Signal | Volatility Regime | Suggested Spread |
|:--|:--|:--|
| **Bullish** | Low / Moderate Vol | Bull Call Debit Spread |
| **Bullish** | Rising Vol | Short Put Credit Spread |
| **Bearish** | Low Vol | Bear Put Debit Spread |
| **Bearish** | High Vol | Short Call Credit Spread |
| **Neutral** | Vol Expansion Expected | Calendar or Diagonal Spread |

### 6. Evaluation
- **Directional metrics:** Accuracy, Precision, Recall, F1, ROC-AUC  
- **Volatility metrics:** RMSE / MAE of realised-vol forecasts  
- **Combined:** Payoff simulation for spread-selection efficiency  

---

## 📊 Results

| Model | Accuracy | F1 Score | RMSE (Vol Forecast) | Comment |
|:--|--:|--:|--:|:--|
| **XGBoost** | 0.63 | 0.61 | — | Strong on non-linear patterns |
| **LSTM** | 0.65 | 0.63 | — | Captures sequential dynamics |
| **Volatility XGBoost** | — | — | 0.00071 | Improved ~14% vs HAR |
| **Hybrid (XGBoost + LSTM + Vol)** | **0.68** | **0.66** | **0.00069** | Best combined performance |

✅ The hybrid model improved trade-direction accuracy by **~7%** and enhanced volatility-aware spread selection, leading to better **reward-to-risk ratios** in back-tests.

---

## 🧰 Tech Stack

**Python** • pandas • NumPy • scikit-learn • xgboost • tensorflow / keras • matplotlib • seaborn • yfinance • ta  

---

## 🚀 Next Steps

- Deploy daily prediction pipeline via **Alpaca API** for live signal generation.  
- Add **Bayesian weighting** to dynamically combine model confidences.  
- Integrate **transaction-cost and slippage** simulation for realistic P&L analysis.  
- Extend to multi-asset universe and **implied-volatility surface forecasting**.

---
