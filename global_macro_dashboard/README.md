# 💹 Global Macro Dashboard — Multi-Asset Economic Data Visualizer (R Shiny)

This project builds an **interactive macroeconomic dashboard** in **R Shiny** that visualises cross-asset relationships across **global markets**, **monetary policy**, and **economic indicators**.  
It integrates data from **FRED**, **Yahoo Finance**, and **Quandl** into a unified macro dataset, transforming it into a **dynamic analytical interface** for macro-financial insight.

---

## 🎯 Objective

Provide a **real-time, multi-asset visualisation system** that enables users to:

- Monitor **macro trends** such as yield-curve shifts, credit spreads, and equity-bond correlations.  
- Compare and scale variables (e.g. GDP Growth vs VIX vs 10-Year Yields).  
- Analyse the **monetary-policy transmission chain** across FX, bonds, and commodities.  
- Detect periods of **market stress**, **policy tightening**, or **volatility compression**.  

---

## ⚙️ Methodology

### 1. Data Collection — `Macro_Data_Collector.R`
Builds a comprehensive **daily macroeconomic dataset (`Daily.csv`)** by connecting to **FRED** and **Yahoo Finance** APIs.

| Data Category | Examples | Source |
|:--|:--|:--|
| **Macroeconomic** | GDP (Global, US, China, Japan, Germany, UK, India), CPI, PCE, Unemployment, M1–M3 | FRED |
| **Equity Indices** | S&P 500, NASDAQ, FTSE 100, DAX, Nikkei 225, Hang Seng, CAC 40 | Yahoo Finance |
| **FX Rates** | EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD | Yahoo Finance / FRED |
| **Commodities** | WTI, Brent, Gold, Silver, Copper | Yahoo Finance |
| **Interest Rates & Yields** | Fed Funds, Discount, Prime, IORB, ON RRP, SOFR, 1M–30Y Treasuries | FRED |
| **Credit Spreads** | ICE BofA AAA, BBB, HY, EM Corporate, EM High Yield | FRED |
| **Inflation Metrics** | Median CPI, Sticky CPI, PCE, 5Y & 10Y Breakevens | FRED |
| **Volatility Index** | VIX CLS | FRED |
| **Real Estate** | Shiller Home Price Index, BIS Residential & Commercial Property Prices | FRED / BIS |

**Processing Steps**

- Merge all datasets into a single **daily-frequency** table.  
- Forward-fill lower-frequency (monthly / quarterly) data using `dplyr::fill()`.  
- Convert all series to numeric format and export to:  
  ```
  C:/Users/pears/OneDrive/Desktop/MACRO/Daily.csv
  ```

---

### 2. Data Transformation — `Macro_Dashboard.R`
Imports the compiled dataset and prepares it for interactive visualisation.

- Parses and converts date fields.  
- Automatically determines **earliest and latest available dates**.  
- Computes **z-scores** for standardised comparison across variables:  
  ```
  z = (x - mean(x)) / sd(x)
  ```
- Scales yield-curve data and builds long-format structures for plotting.  
- Generates a variable list dynamically for user-selectable charts.  

---

### 3. Dashboard Architecture

| Module | Description |
|:--|:--|
| **Dynamic Value Boxes** | Display live values for equities, yields, FX pairs, and commodities |
| **Customisable Line Charts** | Plot up to five user-selected macro variables over custom ranges |
| **Yield-Curve Visualizer** | Display the latest U.S. Treasury curve (3M–20Y) |
| **Z-Score Comparison Chart** | Overlay scaled macro variables to reveal co-movements |
| **Collapsible Panels** | Organised summaries for Equities, Yields, FX, Commodities |
| **Responsive Dark UI** | Symmetrical ribbons, Plotly interactivity, and modern minimal styling |

---

## 📊 Example Output

| Panel | Metric | Example Output |
|:--|:--|:--|
| **Indices** | S&P 500, NASDAQ, FTSE 100, Nikkei 225 | Live closing prices |
| **Yields** | 3M, 2Y, 5Y, 10Y, 20Y | % values via value boxes |
| **FX** | EUR/USD, GBP/USD, USD/JPY, USD/CHF | Spot rates |
| **Commodities** | Gold, Silver, Brent, WTI, Copper | Latest prices (USD) |

---

## 🧠 Core Technologies

**R Packages Used**  
`quantmod` • `FredR` • `data.table` • `dplyr` • `tidyr` • `pipeR` • `shiny` • `shinydashboard` • `ggplot2` • `plotly` • `scales`

---

## 🗂️ Project Structure

```
/Global_Macro_Dashboard
│
├── Macro_Data_Collector.R     # Downloads & merges macro, FX, equity, bond & commodity data
├── Macro_Dashboard.R          # Shiny app for interactive macro visualisation
├── Daily.csv                  # Generated unified dataset (auto-output)
└── README.md                  # Project documentation
```

---

## 🚀 Usage

1️⃣ Run **`Macro_Data_Collector.R`** to build and save the dataset.  
2️⃣ Run **`Macro_Dashboard.R`** to launch the interactive dashboard:  
```r
shiny::runApp("Macro_Dashboard.R")
```
3️⃣ Explore global macro trends interactively — yields, FX, equities, inflation, and commodities in one unified view.

---

## 🌍 Future Enhancements

- Add **inflation-by-country** map visualisation.  
- Include **correlation heatmaps** and **macro regime clustering**.  
- Integrate **policy-rate differentials** and **volatility regime detection**.  
- Automate daily data refresh via `taskscheduleR` or CRON scheduling.  

---
