library(quantmod)
library(dplyr)
library(pipeR)
library(tidyr)
library(tibble)
library(data.table)


devtools::install_github("jcizel/FredR")

api.key = .....

fred <- FredR::FredR(api.key)

#Search results for FRED
#gdp.series <- fred$series.search("GDP")

#GROSS DOMESTIC PRODUCT ####

GDP_World <- fred$series.observations(series_id = 'NYGDPMKTPCDWLD')
GDP_US <- fred$series.observations(series_id = 'GDPC1')
GDP_China <- fred$series.observations(series_id = 'MKTGDPCNA646NWDB')
GDP_Japan <- fred$series.observations(series_id = 'JPNNGDP')
GDP_Germany <- fred$series.observations(series_id = 'CLVMNACSCAB1GQDE')
GDP_India <- fred$series.observations(series_id = 'NGDPRNSAXDCINQ')
GDP_UK <- fred$series.observations(series_id = 'UKNGDP')
GDP_France <- fred$series.observations(series_id = 'CLVMNACSCAB1GQFR')

#INDICES ####

#S&P, Dow, Nasdaq, Russell, FTSE, Hang Seng, Nikkei, CAC, DAX

symbolBasket_Indices <- c('^GSPC','^DJI', '^IXIC', '^RUT', '^FTSE', '^HSI', '^N225', '^FCHI', '^GDAXI')
getSymbols(symbolBasket_Indices , src='yahoo', from = "1900-01-01")

S_and_P_Daily <- as.data.frame(GSPC)
S_and_P_Daily <- setDT(S_and_P_Daily, keep.rownames = TRUE)[]   
Dow_Daily <- as.data.frame(DJI)
Dow_Daily <- setDT(Dow_Daily, keep.rownames = TRUE)[]  
Nasdaq_Daily <- as.data.frame(IXIC)
Nasdaq_Daily <- setDT(Nasdaq_Daily, keep.rownames = TRUE)[] 
Russell_Daily <- RUT
FTSE_Daily <- as.data.frame(FTSE)
FTSE_Daily <- setDT(FTSE_Daily, keep.rownames = TRUE)[] 
Hang_Seng_Daily <- as.data.frame(HSI)
Hang_Seng_Daily <- setDT(Hang_Seng_Daily, keep.rownames = TRUE)[] 
Nikkei_Daily <- as.data.frame(N225)
Nikkei_Daily <- setDT(Nikkei_Daily, keep.rownames = TRUE)[] 
CAC_Daily <-  FCHI
DAX_Daily <- as.data.frame(GDAXI)
DAX_Daily <- setDT(DAX_Daily, keep.rownames = TRUE)[] 


#FX ####

# EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD

symbolBasket_FX <- c('EURUSD=X','JPY=X', 'GBPUSD=X', 'CHF=X', 'AUDUSD=X', 'CAD=X', 'NZDUSD=X')
getSymbols(symbolBasket_FX , src='yahoo', from = "1900-01-01")

EURUSD_Yahoo_Daily <- as.data.frame(`EURUSD=X`)
EURUSD_Yahoo_Daily <- setDT(EURUSD_Yahoo_Daily, keep.rownames = TRUE)[] 
USDJPY_Yahoo_Daily <- as.data.frame(`JPY=X`)
USDJPY_Yahoo_Daily <- setDT(USDJPY_Yahoo_Daily, keep.rownames = TRUE)[] 
GBPUSD_Yahoo_Daily <- as.data.frame(`GBPUSD=X`)
GBPUSD_Yahoo_Daily <- setDT(GBPUSD_Yahoo_Daily, keep.rownames = TRUE)[] 
USDCHF_Yahoo_Daily <- as.data.frame(`CHF=X`)
USDCHF_Yahoo_Daily <- setDT(USDCHF_Yahoo_Daily, keep.rownames = TRUE)[] 
AUDUSD_Yahoo_Daily <-as.data.frame(`AUDUSD=X`)
AUDUSD_Yahoo_Daily <- setDT(AUDUSD_Yahoo_Daily, keep.rownames = TRUE)[] 
USDCAD_Yahoo_Daily <-as.data.frame(`CAD=X`)
USDCAD_Yahoo_Daily <- setDT(USDCAD_Yahoo_Daily, keep.rownames = TRUE)[] 
NZDUSD_Yahoo_Daily <-as.data.frame(`NZDUSD=X`)
NZDUSD_Yahoo_Daily <- setDT(NZDUSD_Yahoo_Daily, keep.rownames = TRUE)[] 

#For rates joing back a little further can join FRED data

Nominal_Dollar_Index_Daily <- fred$series.observations(series_id = 'DTWEXBGS')
EURUSD_Daily <- fred$series.observations(series_id = 'DEXUSEU')
GBPUSD_Daily <- fred$series.observations(series_id = 'DEXUSUK')
USDCNY_Daily <- fred$series.observations(series_id = 'DEXCHUS')
USDJPY_Daily <- fred$series.observations(series_id = 'DEXJPUS')
USDINR_Daily <- fred$series.observations(series_id = 'DEXINUS')

# COMMODITIES ####

# Metals and Energy (Yahoo)
symbolBasket_Commodities <- c('GC=F', 'SI=F', 'HG=F', 'CL=F', 'BZ=F')  # Gold, Silver, Copper, WTI, Brent
getSymbols(symbolBasket_Commodities, src = 'yahoo', from = "1900-01-01")

# Convert each to data.table with a Date column
Gold_Daily <- setDT(as.data.frame(`GC=F`), keep.rownames = TRUE)[]
Silver_Daily <- setDT(as.data.frame(`SI=F`), keep.rownames = TRUE)[]
Copper_Daily <- setDT(as.data.frame(`HG=F`), keep.rownames = TRUE)[]
WTI_Daily <- setDT(as.data.frame(`CL=F`), keep.rownames = TRUE)[]
Brent_Daily <- setDT(as.data.frame(`BZ=F`), keep.rownames = TRUE)[]

#REAL ESTATE ####

House_Prices_Shiller_US_Monthly <- fred$series.observations(series_id = 'CSUSHPINSA')
Real_Residential_Property_Prices_US_Quarterly <- fred$series.observations(series_id = 'QUSR368BIS')
Real_Residential_Property_Prices_China_Quarterly <- fred$series.observations(series_id = 'QCNR368BIS')
Real_Residential_Property_Prices_Japan_Quarterly <- fred$series.observations(series_id = 'QJPR368BIS')
Real_Residential_Property_Prices_Germany_Quarterly <- fred$series.observations(series_id = 'QDER368BIS')
Real_Residential_Property_Prices_India_Quarterly <- fred$series.observations(series_id = 'QINR368BIS')
Real_Residential_Property_Prices_UK_Quarterly <- fred$series.observations(series_id = 'QGBR368BIS')
Real_Residential_Property_Prices_France_Quarterly <- fred$series.observations(series_id = 'QFRR368BIS')

Commercial_Real_Estate_Prices_US_Quarterly <- fred$series.observations(series_id = 'COMREPUSQ159N')
Commercial_Real_Estate_Prices_UK_Quarterly <- fred$series.observations(series_id = 'COMREPGBQ159N')

#EMPLOYMENT ####

#The unemployment rate is widely used to assess the state of the economy because it varies strongly with the business cycle

Unemployment_Rate_US_Monthly <- fred$series.observations(series_id = 'UNRATE')
Unemployment_Rate_UK_Monthly <- fred$series.observations(series_id = 'AURUKM')

#MONEY SUPPLY ####

M1_US <- fred$series.observations(series_id = 'WM1NS')
M2_US <- fred$series.observations(series_id = 'WM2NS')
M3_US <- fred$series.observations(series_id = 'MABMM301USM189S')

#VIX ####
#measures market expectation of near term volatility conveyed by stock index option prices

VIX_Daily <- fred$series.observations(series_id = 'VIXCLS')

#INFLATION ####

#Split between manufacturing and services

Median_CPI_US_Monthly <- fred$series.observations(series_id = 'MEDCPIM158SFRBCLE')

Sticky_Price_CPI_US_Monthly <- fred$series.observations(series_id = 'CORESTICKM159SFRBATL')

PCE_US_Monthly <- fred$series.observations(series_id = 'PCE')

Five_Year_Breakeven_US_Daily <- fred$series.observations(series_id = 'T5YIE')

Ten_Year_Breakeven_US_Daily <- fred$series.observations(series_id = 'T10YIE')

#INTEREST RATES ####

#DISCOUNT RATE - rate at which commercial banks can borrow directly from their central bank, below is Discount Window Primary Credit Rate
Discount_Rate_US_Daily <- fred$series.observations(series_id = 'DPCREDIT')

#PRIME, the lending rate provided by commercial banks to their highest quality corporate customers, and serves as the benchmark for rates provided for other loans
Prime_Rate_US_Daily <- fred$series.observations(series_id = 'DPRIME')

#EFFR, This target is the rate at which commercial banks borrow and lend their excess reserves to each other overnight.
Effective_Federal_Funds_Rate_Monthly <- fred$series.observations(series_id = 'FEDFUNDS')

#the IORB rate is the return rate depository institutions can earn for holding reserves at the Federal Reserve. Meanwhile, the EFFR represents the cost of interbank borrowing and is determined by the market. The difference between these two rates present arbitrage opportunities for banks.
IR_on_Reserve_Balances_Daily <- fred$series.observations(series_id = 'IORB')

#The overnight reverse repurchase agreement (ON RRP) rate is the interest rate that a broad set of financial institutions can earn on deposits with the Fed.
Overnight_Repo_US <- fred$series.observations(series_id = 'RRPONTSYD')

#Secured Overnight Financing Rate
SOFR_Daily <- fred$series.observations(series_id = 'SOFR')

# Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations
REPO_Daily <- fred$series.observations(series_id = 'RRPONTSYD')

#YIELD CURVE ####
# bond yields are a proxy for expectations for the federal funds rate in X years' time

#Inversion measure, recession indicator
Ten_Year_Minus_Two_Year_Daily <- fred$series.observations(series_id = 'T10Y2Y')
Ten_Year_Minus_Three_Month_Daily <- fred$series.observations(series_id = 'T10Y3M')
Ten_Year_Minus_FFR_Daily <- fred$series.observations(series_id = 'T10YFF')

#Constant Maturity yields, The X-year constant maturity Treasury (CMT) represents the X-year yield of the most recently auctioned Treasury securities.
One_Month_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS1MO')
Three_Month_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS3MO')
Six_Month_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS6MO')
One_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS1')
Two_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS2')
Five_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS5')
Ten_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS10')
Twenty_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS20')
Thirty_Year_Constant_Yield_Daily <- fred$series.observations(series_id = 'DGS30')

Yield_Curve <- data.frame()

One_Month_Real_IR_Monthly <- fred$series.observations(series_id = 'REAINTRATREARAT1MO')

One_Year_Real_IR_Monthly <- fred$series.observations(series_id = 'REAINTRATREARAT1YE')

Ten_Year_Real_IR_Monthly <- fred$series.observations(series_id = 'REAINTRATREARAT10Y')

#BOND INDICES ####

#ICE BofA US Index Option-Adjusted Spread

High_Yield_US_Daily <- fred$series.observations(series_id = 'BAMLH0A0HYM2')
AAA_US_Daily <- fred$series.observations(series_id = 'BAMLC0A1CAAA')
BBB_US_Daily <- fred$series.observations(series_id = 'BAMLC0A4CBBB')

Moodys_Seasoned_Aaa_Daily <- fred$series.observations(series_id = 'DAAA')

Emerging_market_corporate_Daily <- fred$series.observations(series_id = 'BAMLEMCBPIOAS')
Emerging_market_high_yield_Daily <- fred$series.observations(series_id = 'BAMLEMHBHYCRPIOAS')

High_Yield_EURO_Daily <- fred$series.observations(series_id = 'BAMLHE00EHYIOAS')

#Daily Data ####
today_date <- Sys.Date()
Date <- seq(as.Date("1960-01-01"),as.Date(today_date),by = 1)

Daily_Dataframe <- data.frame(Date)

#GDP
Daily_Dataframe$GDP_US <- GDP_US[match(Daily_Dataframe$Date, GDP_US$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(GDP_US)
Daily_Dataframe$GDP_US <- as.numeric(unlist(Daily_Dataframe$GDP_US)) 

#Stock Market Indices
Daily_Dataframe$S_and_P_Close <- S_and_P_Daily[match(Daily_Dataframe$Date, S_and_P_Daily$rn), "GSPC.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(S_and_P_Close)
Daily_Dataframe$S_and_P_Close <- as.numeric(unlist(Daily_Dataframe$S_and_P_Close)) 

Daily_Dataframe$Nasdaq_Close <- Nasdaq_Daily[match(Daily_Dataframe$Date, Nasdaq_Daily$rn), "IXIC.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Nasdaq_Close)
Daily_Dataframe$Nasdaq_Close <- as.numeric(unlist(Daily_Dataframe$Nasdaq_Close)) 

Daily_Dataframe$Hang_Seng_Close <- Hang_Seng_Daily[match(Daily_Dataframe$Date, Hang_Seng_Daily$rn), "HSI.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Hang_Seng_Close)
Daily_Dataframe$Hang_Seng_Close <- as.numeric(unlist(Daily_Dataframe$Hang_Seng_Close)) 

Daily_Dataframe$Nikkei_Close <- Nikkei_Daily[match(Daily_Dataframe$Date, Nikkei_Daily$rn), "N225.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Nikkei_Close)
Daily_Dataframe$Nikkei_Close <- as.numeric(unlist(Daily_Dataframe$Nikkei_Close)) 

Daily_Dataframe$Dax_Close <- DAX_Daily[match(Daily_Dataframe$Date, DAX_Daily$rn), "GDAXI.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Dax_Close)
Daily_Dataframe$Dax_Close <- as.numeric(unlist(Daily_Dataframe$Dax_Close)) 

Daily_Dataframe$FTSE_Close <- FTSE_Daily[match(Daily_Dataframe$Date, FTSE_Daily$rn), "FTSE.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(FTSE_Close)
Daily_Dataframe$FTSE_Close <- as.numeric(unlist(Daily_Dataframe$FTSE_Close)) 

#Bond Market Indices

Daily_Dataframe$AAA_Yield_US <-  AAA_US_Daily[match(Daily_Dataframe$Date, AAA_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(AAA_Yield_US)
Daily_Dataframe$AAA_Yield_US <- as.numeric(unlist(Daily_Dataframe$AAA_Yield_US)) 

Daily_Dataframe$BBB_Yield_US <-  BBB_US_Daily[match(Daily_Dataframe$Date, BBB_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(BBB_Yield_US)
Daily_Dataframe$BBB_Yield_US <- as.numeric(unlist(Daily_Dataframe$BBB_Yield_US)) 

Daily_Dataframe$High_Yield_US <-  High_Yield_US_Daily[match(Daily_Dataframe$Date, High_Yield_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(High_Yield_US)
Daily_Dataframe$High_Yield_US <- as.numeric(unlist(Daily_Dataframe$High_Yield_US)) 

Daily_Dataframe$Emerging_Corporate_US <-  Emerging_market_corporate_Daily[match(Daily_Dataframe$Date, Emerging_market_corporate_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Emerging_Corporate_US)
Daily_Dataframe$Emerging_Corporate_US <- as.numeric(unlist(Daily_Dataframe$Emerging_Corporate_US)) 

Daily_Dataframe$Emerging_High_Yield_US <-  Emerging_market_high_yield_Daily[match(Daily_Dataframe$Date, Emerging_market_high_yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Emerging_High_Yield_US)
Daily_Dataframe$Emerging_High_Yield_US <- as.numeric(unlist(Daily_Dataframe$Emerging_High_Yield_US)) 

#FX, 4 majors

Daily_Dataframe$EURUSD_Close <- EURUSD_Yahoo_Daily[match(Daily_Dataframe$Date, EURUSD_Yahoo_Daily$rn), "EURUSD=X.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(EURUSD_Close)
Daily_Dataframe$EURUSD_Close <- as.numeric(unlist(Daily_Dataframe$EURUSD_Close)) 

Daily_Dataframe$USDJPY_Close <- USDJPY_Yahoo_Daily[match(Daily_Dataframe$Date, USDJPY_Yahoo_Daily$rn), "JPY=X.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(USDJPY_Close)
Daily_Dataframe$USDJPY_Close <- as.numeric(unlist(Daily_Dataframe$USDJPY_Close)) 

Daily_Dataframe$GBPUSD_Close <- GBPUSD_Yahoo_Daily[match(Daily_Dataframe$Date, GBPUSD_Yahoo_Daily$rn), "GBPUSD=X.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(GBPUSD_Close)
Daily_Dataframe$GBPUSD_Close <- as.numeric(unlist(Daily_Dataframe$GBPUSD_Close)) 

Daily_Dataframe$USDCHF_Close <- USDCHF_Yahoo_Daily[match(Daily_Dataframe$Date, USDCHF_Yahoo_Daily$rn), "CHF=X.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(USDCHF_Close)
Daily_Dataframe$USDCHF_Close <- as.numeric(unlist(Daily_Dataframe$USDCHF_Close)) 

#Commodities

Daily_Dataframe$Brent <- Brent_Daily[match(Daily_Dataframe$Date, Brent_Daily$rn), "BZ=F.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Brent)
Daily_Dataframe$Brent <- as.numeric(unlist(Daily_Dataframe$Brent))

Daily_Dataframe$WTI <- WTI_Daily[match(Daily_Dataframe$Date, WTI_Daily$rn), "CL=F.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(WTI)
Daily_Dataframe$WTI <- as.numeric(unlist(Daily_Dataframe$WTI))

Daily_Dataframe$Gold <- Gold_Daily[match(Daily_Dataframe$Date, Gold_Daily$rn), "GC=F.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Gold)
Daily_Dataframe$Gold <- as.numeric(unlist(Daily_Dataframe$Gold))

Daily_Dataframe$Silver <- Silver_Daily[match(Daily_Dataframe$Date, Silver_Daily$rn), "SI=F.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Silver)
Daily_Dataframe$Silver <- as.numeric(unlist(Daily_Dataframe$Silver))

Daily_Dataframe$Copper <- Copper_Daily[match(Daily_Dataframe$Date, Copper_Daily$rn), "HG=F.Close"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Copper)
Daily_Dataframe$Copper <- as.numeric(unlist(Daily_Dataframe$Copper))

#Real Estate

Daily_Dataframe$Shiller_US_House <-  House_Prices_Shiller_US_Monthly[match(Daily_Dataframe$Date, House_Prices_Shiller_US_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Shiller_US_House)
Daily_Dataframe$Shiller_US_House <- as.numeric(unlist(Daily_Dataframe$Shiller_US_House)) 

#Unemployment 

Daily_Dataframe$Unemployment_US <-  Unemployment_Rate_US_Monthly[match(Daily_Dataframe$Date, Unemployment_Rate_US_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Unemployment_US)
Daily_Dataframe$Unemployment_US <- as.numeric(unlist(Daily_Dataframe$Unemployment_US)) 

Daily_Dataframe$Unemployment_UK <-  Unemployment_Rate_UK_Monthly[match(Daily_Dataframe$Date, Unemployment_Rate_UK_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Unemployment_UK)
Daily_Dataframe$Unemployment_UK <- as.numeric(unlist(Daily_Dataframe$Unemployment_UK)) 

#Money Supply

Daily_Dataframe$M1_US <-  M1_US[match(Daily_Dataframe$Date, M1_US$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(M1_US)
Daily_Dataframe$M1_US <- as.numeric(unlist(Daily_Dataframe$M1_US)) 

Daily_Dataframe$M2_US <-  M2_US[match(Daily_Dataframe$Date, M2_US$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(M2_US)
Daily_Dataframe$M2_US <- as.numeric(unlist(Daily_Dataframe$M2_US)) 

Daily_Dataframe$M3_US <-  M3_US[match(Daily_Dataframe$Date, M3_US$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(M3_US)
Daily_Dataframe$M3_US <- as.numeric(unlist(Daily_Dataframe$M3_US)) 

#Vix

Daily_Dataframe$VIX <-  VIX_Daily[match(Daily_Dataframe$Date, VIX_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(VIX)
Daily_Dataframe$VIX <- as.numeric(unlist(Daily_Dataframe$VIX)) 

#Inflation

Daily_Dataframe$Median_CPI_US <-  Median_CPI_US_Monthly[match(Daily_Dataframe$Date, Median_CPI_US_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Median_CPI_US)
Daily_Dataframe$Median_CPI_US <- as.numeric(unlist(Daily_Dataframe$Median_CPI_US)) 

Daily_Dataframe$Sticky_Price_CPI_US <-  Sticky_Price_CPI_US_Monthly[match(Daily_Dataframe$Date, Sticky_Price_CPI_US_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Sticky_Price_CPI_US)
Daily_Dataframe$Sticky_Price_CPI_US <- as.numeric(unlist(Daily_Dataframe$Sticky_Price_CPI_US)) 

Daily_Dataframe$PCE_US <-  PCE_US_Monthly[match(Daily_Dataframe$Date, PCE_US_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(PCE_US)
Daily_Dataframe$PCE_US <- as.numeric(unlist(Daily_Dataframe$PCE_US)) 

Daily_Dataframe$Five_Year_Breakeven <-  Five_Year_Breakeven_US_Daily[match(Daily_Dataframe$Date, Five_Year_Breakeven_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Five_Year_Breakeven)
Daily_Dataframe$Five_Year_Breakeven <- as.numeric(unlist(Daily_Dataframe$Five_Year_Breakeven)) 

Daily_Dataframe$Ten_Year_Breakeven <-  Ten_Year_Breakeven_US_Daily[match(Daily_Dataframe$Date, Ten_Year_Breakeven_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Breakeven)
Daily_Dataframe$Ten_Year_Breakeven <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Breakeven)) 

#Interest Rates

Daily_Dataframe$EFFR_US <-  Effective_Federal_Funds_Rate_Monthly[match(Daily_Dataframe$Date, Effective_Federal_Funds_Rate_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(EFFR_US)
Daily_Dataframe$EFFR_US <- as.numeric(unlist(Daily_Dataframe$EFFR_US))

Daily_Dataframe$Discount_Rate_US <-  Discount_Rate_US_Daily[match(Daily_Dataframe$Date, Discount_Rate_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Discount_Rate_US)
Daily_Dataframe$Discount_Rate_US <- as.numeric(unlist(Daily_Dataframe$Discount_Rate_US))

Daily_Dataframe$Prime_Rate_US <-  Prime_Rate_US_Daily[match(Daily_Dataframe$Date, Prime_Rate_US_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Prime_Rate_US)
Daily_Dataframe$Prime_Rate_US <- as.numeric(unlist(Daily_Dataframe$Prime_Rate_US))

Daily_Dataframe$IORB_US <-  IR_on_Reserve_Balances_Daily[match(Daily_Dataframe$Date, IR_on_Reserve_Balances_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(IORB_US)
Daily_Dataframe$IORB_US <- as.numeric(unlist(Daily_Dataframe$IORB_US))

Daily_Dataframe$ON_RRP <-  Overnight_Repo_US[match(Daily_Dataframe$Date, Overnight_Repo_US$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(ON_RRP)
Daily_Dataframe$ON_RRP <- as.numeric(unlist(Daily_Dataframe$ON_RRP))

#Yield Curve

Daily_Dataframe$Ten_Year_Minus_FFR <-  Ten_Year_Minus_FFR_Daily[match(Daily_Dataframe$Date, Ten_Year_Minus_FFR_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Minus_FFR)
Daily_Dataframe$Ten_Year_Minus_FFR <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Minus_FFR))

Daily_Dataframe$Ten_Year_Minus_Three_Month <-  Ten_Year_Minus_Three_Month_Daily[match(Daily_Dataframe$Date, Ten_Year_Minus_Three_Month_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Minus_Three_Month)
Daily_Dataframe$Ten_Year_Minus_Three_Month <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Minus_Three_Month))

Daily_Dataframe$Ten_Year_Minus_Two_Year_Daily <-  Ten_Year_Minus_Two_Year_Daily[match(Daily_Dataframe$Date, Ten_Year_Minus_Two_Year_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Minus_Two_Year_Daily)
Daily_Dataframe$Ten_Year_Minus_Two_Year_Daily <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Minus_Two_Year_Daily))

Daily_Dataframe$One_Month_Yield <-  One_Month_Constant_Yield_Daily[match(Daily_Dataframe$Date, One_Month_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(One_Month_Yield)
Daily_Dataframe$One_Month_Yield <- as.numeric(unlist(Daily_Dataframe$One_Month_Yield))

Daily_Dataframe$Three_Month_Yield <-  Three_Month_Constant_Yield_Daily[match(Daily_Dataframe$Date, Three_Month_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Three_Month_Yield)
Daily_Dataframe$Three_Month_Yield <- as.numeric(unlist(Daily_Dataframe$Three_Month_Yield))

Daily_Dataframe$Six_Month_Yield <-  Six_Month_Constant_Yield_Daily[match(Daily_Dataframe$Date, Six_Month_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Six_Month_Yield)
Daily_Dataframe$Six_Month_Yield <- as.numeric(unlist(Daily_Dataframe$Six_Month_Yield))

Daily_Dataframe$One_Year_Yield <-  One_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, One_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(One_Year_Yield)
Daily_Dataframe$One_Year_Yield <- as.numeric(unlist(Daily_Dataframe$One_Year_Yield))

Daily_Dataframe$Two_Year_Yield <-  Two_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, Two_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Two_Year_Yield)
Daily_Dataframe$Two_Year_Yield <- as.numeric(unlist(Daily_Dataframe$Two_Year_Yield))

Daily_Dataframe$Five_Year_Yield <-  Five_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, Five_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Five_Year_Yield)
Daily_Dataframe$Five_Year_Yield <- as.numeric(unlist(Daily_Dataframe$Five_Year_Yield))

Daily_Dataframe$Ten_Year_Yield <-  Ten_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, Ten_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Yield)
Daily_Dataframe$Ten_Year_Yield <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Yield))

Daily_Dataframe$Twenty_Year_Yield <-  Twenty_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, Twenty_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Twenty_Year_Yield)
Daily_Dataframe$Twenty_Year_Yield <- as.numeric(unlist(Daily_Dataframe$Twenty_Year_Yield))

Daily_Dataframe$Thirty_Year_Yield <-  Thirty_Year_Constant_Yield_Daily[match(Daily_Dataframe$Date, Thirty_Year_Constant_Yield_Daily$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Thirty_Year_Yield)
Daily_Dataframe$Thirty_Year_Yield <- as.numeric(unlist(Daily_Dataframe$Thirty_Year_Yield))

Daily_Dataframe$One_Month_Real_IR <-  One_Month_Real_IR_Monthly[match(Daily_Dataframe$Date, One_Month_Real_IR_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(One_Month_Real_IR)
Daily_Dataframe$One_Month_Real_IR <- as.numeric(unlist(Daily_Dataframe$One_Month_Real_IR))

Daily_Dataframe$One_Year_Real_IR <-  One_Year_Real_IR_Monthly[match(Daily_Dataframe$Date, One_Year_Real_IR_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(One_Year_Real_IR)
Daily_Dataframe$One_Year_Real_IR <- as.numeric(unlist(Daily_Dataframe$One_Year_Real_IR))

Daily_Dataframe$Ten_Year_Real_IR <-  Ten_Year_Real_IR_Monthly[match(Daily_Dataframe$Date, Ten_Year_Real_IR_Monthly$date), "value"]
Daily_Dataframe <- Daily_Dataframe %>% fill(Ten_Year_Real_IR)
Daily_Dataframe$Ten_Year_Real_IR <- as.numeric(unlist(Daily_Dataframe$Ten_Year_Real_IR))


fwrite(Daily_Dataframe, "C:\\Users\\pears\\OneDrive\\Desktop\\MACRO\\Daily.csv")


