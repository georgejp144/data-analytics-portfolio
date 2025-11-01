💹 Global Macro Dashboard — Multi-Asset Economic Data Visualizer (R Shiny)

This project builds an interactive macroeconomic dashboard in R Shiny that visualises cross-asset relationships across global markets, monetary policy, and economic indicators.
It integrates data from FRED, Yahoo Finance, and Quandl into a unified macro dataset, transforming it into a dynamic analytics interface for macro-financial insights.

🎯 Objective

To provide a real-time, multi-asset visualisation system that enables users to:

Monitor macro trends such as yield-curve shifts, credit spreads, and equity-bond correlations.

Compare and scale any variables (e.g. GDP Growth vs VIX vs 10-Year Yields).

Inspect the monetary-policy transmission chain across FX, bonds, and commodities.

Detect periods of market stress, policy tightening, or volatility compression.

⚙️ Methodology
1. Data Aggregation — Macro_Data_Collector.R

This script builds the foundation of the dashboard by connecting to FRED and Yahoo Finance APIs to construct a comprehensive daily macroeconomic dataset (Daily.csv).

Pulls global and domestic GDP, CPI, Unemployment, and Money Supply (M1–M3).

Imports equity indices (S&P 500, NASDAQ, FTSE, DAX, Nikkei, Hang Seng, etc.).

Gathers FX rates (EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD).

Adds commodities (WTI, Brent, Gold, Silver, Copper).

Includes interest rates, credit spreads, and yield curve data (1M–30Y).

Captures VIX, breakeven inflation, and policy rates such as Fed Funds, Prime, and IORB.

Uses dplyr::fill() to forward-fill monthly or quarterly series for daily alignment.

Outputs the cleaned, unified file as:

C:/Users/pears/OneDrive/Desktop/MACRO/Daily.csv

2. Dashboard Visualisation — Macro_Dashboard.R

This Shiny app reads the compiled Daily.csv and transforms it into an interactive multi-panel dashboard with Plotly visuals and live summary tiles.

Automatically determines earliest and latest date ranges.

Computes z-scores for all variables for comparability:

𝑧
=
𝑥
−
𝑥
ˉ
𝑠
z=
s
x−
x
ˉ
	​


Builds a minimalist, symmetrical dark-themed UI using shinydashboard.

Enables variable selection across multiple modules for custom comparisons.

3. Dashboard Features
Module	Description
Dynamic Value Boxes	Display live values for equities, yields, FX, and commodities
Customisable Line Charts	Plot up to five variables over custom date ranges
Yield Curve Visualiser	Plot the latest U.S. Treasury curve (3M–20Y) dynamically
Z-Score Comparison Chart	Overlay scaled macro variables to detect co-movements
Collapsible Panels	Organised market summaries: Indices, Yields, FX, Commodities
Responsive UI Design	Clean dark-ribbon interface, interactive Plotly visuals
🧠 Core Technologies

R Packages Used:
quantmod • FredR • data.table • dplyr • tidyr • pipeR • shiny • shinydashboard • ggplot2 • plotly • scales
