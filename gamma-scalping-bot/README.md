# 🧠 Gamma Scalping & Volatility Arbitrage Bot

This project demonstrates a delta-neutral options trading strategy using the Alpaca API.
The bot automatically monitors option Greeks (Delta, Gamma) and dynamically hedges the underlying stock
to capture profits from volatility — a process known as **gamma scalping**.

---

## ⚙️ How It Works
1. Connects to Alpaca API using secure keys in `.env`
2. Fetches the option chain for a given underlying (e.g., JPM)
3. Calculates Greeks (Delta, Gamma) using Black–Scholes
4. Maintains delta neutrality by trading the underlying
5. Rebalances periodically to capture volatility (gamma scalping)

---

## 🧰 Tech Stack
Python • Alpaca API • pandas • SciPy • asyncio • numpy

---

📬 **Author:** George Pearson  
[Data Analyst]  
[LinkedIn](https://www.linkedin.com/in/george-pearson-938914287/)
