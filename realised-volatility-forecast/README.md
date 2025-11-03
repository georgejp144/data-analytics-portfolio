📈 Realised Volatility Forecast — HAR + XGBoost Model

This project builds a machine-learning volatility forecasting engine that predicts the next 14-day realised volatility for the QQQ ETF (NASDAQ-100).
It combines econometric memory (HAR model) with nonlinear ML power (XGBoost) and technical indicators to improve short-term volatility forecasts used in options trading, vol-timing, or risk-management strategies.


⚙️ How It Works

Data Collection

- Connects securely to the Alpaca Data API using credentials stored in .env.

- Downloads ~3 years of daily OHLCV data for QQQ.

Feature Engineering

- Computes log returns and realised volatility (14-day, annualised).

- Builds HAR-style lag features — 1-, 5-, 22-day volatility memory.

- Adds technical indicators: RSI (14), Bollinger Band width, ATR (14), return-volatility ratios, and volume z-scores.

Target Construction

- Shifts realised volatility 14 days forward → future target variable.

Model Training

- Scales features with RobustScaler.

- Performs TimeSeriesSplit cross-validation and GridSearchCV to tune XGBoost hyperparameters.

Evaluation & Forecasting

- Reports MAE, RMSE, and Relative MAE on out-of-sample data.

- Generates a live forecast for the next 14 days’ annualised volatility.

Deployment Artifacts

- Saves the tuned model (rv_har_xgb_tuned_model.joblib), scaler (rv_feature_scaler.joblib), and training dataset (rv_training_data.csv).


🧰 Tech Stack

Data:	Alpaca API • pandas • numpy • datetime
ML & Validation:	scikit-learn (GridSearchCV, TimeSeriesSplit) • xgboost
Feature Engineering:	ta (RSI, Bollinger Bands, ATR)
Evaluation & Persistence:	MAE / RMSE metrics • joblib
Environment:	dotenv (for API keys) • Python 3.10+


📊 Outputs

MAE:	Mean Absolute Error between predicted vs actual volatility
RMSE:	Root Mean Squared Error – sensitive to outliers
Relative MAE:	Model error as % of average realised vol
Forecast:	Predicted next 14-day annualised realised volatility
