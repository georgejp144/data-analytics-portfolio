# 📈 Realised Volatility Forecast — HAR & XGBoost

This project compares econometric (HAR) and machine-learning (XGBoost) approaches to forecasting realised volatility for NASDAQ index options.

---

## 🎯 Objective
Develop and compare models that predict short-term realised volatility using lagged volatility terms and market indicators.

---

## ⚙️ Methodology
1. Aggregate high-frequency prices into daily realised volatility.
2. Build **HAR(1,5,22)** model in R.
3. Train **XGBoost** regression model in Python using engineered features.
4. Compare performance using RMSE, MAE, and directional-accuracy metrics.

---

## 📊 Results
| Model | RMSE | MAE | Directional Accuracy |
|:--|--:|--:|--:|
| HAR | 0.00084 | 0.00063 | 0.56 |
| XGBoost | **0.00072** | **0.00052** | **0.63** |

✅ XGBoost improved accuracy by ~15% over HAR while retaining interpretability through feature importance.

---

## 🧰 Tech Stack
Python • R • pandas • NumPy • scikit-learn • xgboost • matplotlib  

---

## 📸 Preview (optional)
_Add a chart later if you have one:_
