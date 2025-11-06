# 📊 Options Swing-Trading Indicator — Gamma Scalping & Volatility Timing Dashboard

<p align="center">
  <img src="assets/Power%20BI%20Dash.jpg" width="900">
</p>

This project builds an **interactive Power BI analytics dashboard** that identifies *high-probability long-volatility entry setups* for **gamma-scalping** and **straddle-based volatility-timing**.  
It combines **14-Day Implied Volatility**, **LSTM-predicted 14-Day Realised Volatility**, **price-compression measures**, and **event-timing signals** to highlight when **volatility is underpriced** and **expansion is likely**.

---

## 🎯 Objective

Identify **optimal entry timing** for **ATM straddles, long gamma, and calendar spreads** on NASDAQ assets (primarily **QQQ**) by detecting when:

- **Implied Volatility is cheap** relative to **forecasted Realised Volatility**
- **Price is coiling** (volatility compression phase)
- A **catalytic event is approaching** (earnings, CPI, FOMC, etc.)

---

## ⚙️ Data Pipeline

- **Historical price data** sourced from **Alpaca API** and **Yahoo Finance**
- **14-Day Realised Volatility Forecast** generated using an **LSTM neural network**
- **14-Day ATM Implied Volatility** sourced from option-chain data
- **Event calendar** merged to compute *Days Until Catalyst*

All data outputs are combined into a single `.csv` file for direct use in Power BI.

---

## 📊 Dashboard Components

| Component | Purpose | Visual Type |
|:--|:--|:--|
| **14D Implied Volatility** | Market pricing of near-term volatility | KPI Card |
| **LSTM 14D RV Forecast** | Model expectation of future actual volatility | KPI Card |
| **Relative MAE** | Model confidence / forecast reliability | KPI Card |
| **Main Trade Signal (IV − RV)** | Detects volatility under/overpricing | KPI Card w/ conditional color |
| **Days to Event** | Identifies proximity to catalyst | KPI Card |

---

## 🔍 Volatility & Price Structure Visuals

| Visual | Description | Insight |
|:--|:--|:--|
| **Volatility Compression Gauge** | Normalised volatility compression (BB/ATR) | Detects coiling phases |
| **Volatility Expansion Gauge (ATR)** | Measures volatility breakout strength | Confirms expansion phases |
| **14D RV vs 14D IV Historical Chart** | Compares realised vs implied volatility | Identifies cheap vs expensive IV |
| **14D Volatility Percentile Rank** | Current vol regime vs history | Normalises current market state |
| **QQQ Close Price Chart** | Market structure + breakout context | Confirms trade timing |

---

## 🧠 Trade Signal Logic

| Condition | Interpretation | Action |
|:--|:--|:--|
| **IV < LSTM RV Forecast** | Options appear cheap | ✅ Consider Long Gamma / Straddle |
| **Compression Gauge Low** | Market coiling | ⏳ Watch for breakout |
| **Days to Event Small** | Catalyst nearby | ⚡ Expect volatility expansion |

When these align → **High-probability Long-Volatility Setup**.

---

## 🧰 Tech Stack

**Power BI** • **DAX** • **Python (LSTM Forecast Model)** • **Alpaca API** • **yfinance** • **CSV**

---

