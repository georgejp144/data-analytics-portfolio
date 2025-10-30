# 🧠 Gamma Scalping & Volatility Arbitrage Bot

This project demonstrates a delta-neutral options trading strategy using the Alpaca API.
The bot automatically monitors option Greeks (Delta, Gamma) and dynamically hedges the underlying stock
to capture profits from volatility — a process known as **gamma scalping**.

---

## ⚙️ How It Works

1. **Connects to Alpaca API** using secure keys in a `.env` file  
2. **Fetches option chain** for an underlying (e.g., JPM)  
3. **Calculates implied volatility & Greeks** using Black-Scholes  
4. **Maintains delta neutrality** by trading the underlying stock  
5. **Hedges periodically** via an async loop (gamma scalping)

---

## 📦 Setup

```bash
pip install -r requirements.txt
