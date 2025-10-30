# 💹 Global Macro Dashboard — Multi-Asset Economic Data Visualizer (R Shiny)

This project builds an **interactive macroeconomic dashboard** in **R Shiny** that visualises cross-asset relationships across **global markets**, **monetary policy**, and **economic indicators**.  
It transforms thousands of time-series (GDP, inflation, FX, yields, commodities, equities, etc.) into an intuitive analytical interface for macro-financial insight.

---

## 🎯 Objective

To provide a **real-time, multi-asset visualisation system** that allows users to:
- Monitor macro trends such as **yield curve shifts**, **credit spreads**, and **equity-bond correlation**.
- Compare and scale any variables (e.g., GDP growth vs VIX vs 10-Year yields).
- Inspect the **monetary policy transmission chain** across FX, bonds, and commodities.
- Identify periods of **market stress**, **policy tightening**, or **volatility compression**.

---

## ⚙️ Methodology

### 1. Data Pipeline
- Imports pre-built **Daily Macro Dataset (`Daily.csv`)** containing:
  - GDP, CPI, PCE, Unemployment, M1–M3  
  - Equity indices (S&P, NASDAQ, FTSE, Nikkei, DAX, etc.)  
  - FX majors (EUR/USD, USD/JPY, GBP/USD, USD/CHF, etc.)  
  - Commodities (WTI, Brent, Gold, Silver, Copper)  
  - Interest rates and yield curve data (1M–30Y)  
  - VIX, breakeven inflation, and credit spreads  
- Source data gathered from **FRED**, **Quandl**, and **Yahoo Finance** (via the `Macro_Data_Collector.R` script).

### 2. Data Processing
- Computes **z-scores** for each macro variable:  
  \[
  z = \frac{x - \bar{x}}{s}
  \]
  to normalise across differing scales.
- Transforms **yield curve data** into a long-format structure for animated term-structure plotting.
- Builds correlation matrices across equities, yields, and volatility indices.

### 3. Dashboard Features
| Module | Description |
|:--|:--|
| **Dynamic Value Boxes** | Display key asset values and macro indicators |
| **Interactive Line Charts** | Compare up to 3 variables (blue/red/green) over custom date ranges |
| **Yield Curve Visualizer** | Slider-controlled term structure evolution |
| **Correlation Heatmap** | Detect macro relationships and market regime clustering |
| **Narrative Text Outputs** | Inline commentary on macroeconomic relationships (e.g., bond-mortgage link, inflation-yield impact) |

---

## 🧠 Core Technologies

**R Packages Used:**
`shiny` • `shinydashboard` • `data.table` • `dplyr` • `ggplot2` • `plotly` • `ggcorrplot` • `scales` • `rlang`

---
