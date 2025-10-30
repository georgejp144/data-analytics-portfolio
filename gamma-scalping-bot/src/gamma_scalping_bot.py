# Import necessary packets

from datetime import datetime, timedelta
import time
import asyncio
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from scipy.stats import norm
from scipy.optimize import brentq
import nest_asyncio
from alpaca.data.historical.option import OptionHistoricalDataClient, OptionLatestQuoteRequest
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.trading.models import TradeUpdate
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import AssetStatus, ContractType, AssetClass

# Apply nest_asyncio to allow running the event loop

nest_asyncio.apply()

###### Logging into Alpaca ######

# Load env
load_dotenv()

#Getting login details from .env
###### Logging into Alpaca ######

# Load env
load_dotenv()

#Getting login details from .env
TRADE_API_KEY = os.getenv('ALPACA_API_KEY_ID', '')
TRADE_API_SECRET = os.getenv('ALPACA_API_SECRET_KEY', '')
paper = True

#Logging in and connecting to data
trading_client = TradingClient(api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET, paper=paper)
trade_update_stream = TradingStream(api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET, paper=paper)
stock_data_client = StockHistoricalDataClient(api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET)
option_data_client = OptionHistoricalDataClient(api_key=TRADE_API_KEY, secret_key=TRADE_API_SECRET)

# Configuration
underlying_symbol = "JPM"
# "Don’t allow the portfolio to behave like more than $500 of stock exposure in either direction." ie defines when to scalp, the threshold
# The trigger for hedging
max_abs_notional_delta = 500
# Better to use a risk free curve for real examples
risk_free_rate = 0.045
positions = {}

# Add underlying symbol to positions list
print(f"Adding {underlying_symbol} to position list")
positions[underlying_symbol] = {'asset_class': 'us_equity', 'position': 0, 'initial_position': 0}

# Set expiration range for options
today = datetime.now().date()
min_expiration = today + timedelta(days=14)
max_expiration = today + timedelta(days=60)

# Get the latest price of the underlying stock
def get_underlying_price(symbol):

    underlying_trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    underlying_trade_response = stock_data_client.get_stock_latest_trade(underlying_trade_request)
    return underlying_trade_response[symbol].price

underlying_price = get_underlying_price(underlying_symbol)
min_strike = round(underlying_price * 1.01, 2)

print(f"{underlying_symbol} price: {underlying_price}")
print(f"Min Expiration: {min_expiration}, Max Expiration: {max_expiration}, Min Strike: {min_strike}")

# Search for option contracts to add to the portfolio
req = GetOptionContractsRequest(
    underlying_symbols=[underlying_symbol],
    status=AssetStatus.ACTIVE,
    expiration_date_gte=min_expiration,
    expiration_date_lte=max_expiration,
    root_symbol=underlying_symbol,
    type=ContractType.CALL,
    strike_price_gte=str(min_strike),
    limit=5,
)

option_chain_list = trading_client.get_option_contracts(req).option_contracts

# Add the first 3 options to the position list
#Essentially, this block prepares the initial set of options contracts to be included in the gamma scalping strategy.
for option in option_chain_list[:3]:
    symbol = option.symbol
    print(f"Adding {symbol} to position list")
    positions[symbol] = {
        'asset_class': 'us_option',
        'underlying_symbol': option.underlying_symbol,
        'expiration_date': pd.Timestamp(option.expiration_date),
        'strike_price': float(option.strike_price),
        'type': option.type,
        'size': float(option.size),
        'position': 1.0,
        'initial_position': 1.0
    }

# Calculate implied volatility using Black Scholes, note: works best for European options
def calculate_implied_volatility(option_price, S, K, T, r, option_type):
    def option_price_diff(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price - option_price

    return brentq(option_price_diff, 1e-6, 1)

# Calculate option Greeks (Delta and Gamma)
def calculate_greeks(option_price, strike_price, expiry, underlying_price, risk_free_rate, option_type):
    T = (expiry - pd.Timestamp.now()).days / 365
    implied_volatility = calculate_implied_volatility(option_price, underlying_price, strike_price, T, risk_free_rate, option_type)
    d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility ** 2) * T) / (implied_volatility * np.sqrt(T))
    d2 = d1 - implied_volatility * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(T))
    return delta, gamma

# Handling trade updates
# It is the bot’s “eyes” into your account letting it know when trades have actually happened.
async def on_trade_updates(data: TradeUpdate):
    symbol = data.order.symbol
    if symbol in positions:
        if data.event in {'fill', 'partial_fill'}:
            side = data.order.side
            qty = data.order.qty
            filled_avg_price = data.order.filled_avg_price
            position_qty = data.position_qty
            print(f"{data.event} event: {side} {qty} {symbol} @ {filled_avg_price}")
            print(f"updating position from {positions[symbol]['position']} to {position_qty}")
            positions[symbol]['position'] = float(position_qty)
#Subscribes to the trade_update_stream using this on_trade_updates function to receive and process real-time trade updates as they occur.
#Essentially a live notifications of trades filling signup
trade_update_stream.subscribe_trade_updates(on_trade_updates)

# Execute initial trades
# Opens your starting positions automatically when the bot launches.
async def initial_trades():
    await asyncio.sleep(5)
    print("executing initial option trades")
    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_option' and pos['initial_position'] != 0:
            side = 'buy' if pos['initial_position'] > 0 else 'sell'
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(pos['initial_position']),
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"Submitting order to {side} {abs(pos['initial_position'])} contracts of {symbol} at market")
            trading_client.submit_order(order_request)

# Maintain delta-neutral strategy
# Automatically keeps your position delta-neutral by buying or selling the underlying stock whenever your total delta gets too large.
def maintain_delta_neutral():
    current_delta = 0.0
    underlying_price = get_underlying_price(underlying_symbol)

    print(f"Current price of {underlying_symbol} is {underlying_price}")

    for symbol, pos in positions.items():
        if pos['asset_class'] == 'us_equity' and symbol == underlying_symbol:
            current_delta += pos['position']
        elif pos['asset_class'] == 'us_option' and pos['underlying_symbol'] == underlying_symbol:
            option_quote_request = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            option_quote = option_data_client.get_option_latest_quote(option_quote_request)[symbol]
            option_quote_mid = (option_quote.bid_price + option_quote.ask_price) / 2

            delta, gamma = calculate_greeks(
                option_price=option_quote_mid,
                strike_price=pos['strike_price'],
                expiry=pos['expiration_date'],
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                option_type=pos['type']
            )

            current_delta += delta * pos['position'] * pos['size']

    adjust_delta(current_delta, underlying_price)

def adjust_delta(current_delta, underlying_price):
    if current_delta * underlying_price > max_abs_notional_delta:
        side = 'sell'
    elif current_delta * underlying_price < -max_abs_notional_delta:
        side = 'buy'
    else:
        return

    qty = abs(round(current_delta,0))
    order_request = MarketOrderRequest(symbol=underlying_symbol, qty=qty, side=side, type='market', time_in_force='day')
    print(f"Submitting {side} order for {qty} shares of {underlying_symbol} at market")
    trading_client.submit_order(order_request)

# Gamma scalping strategy
# Repeatedly checks your delta and hedges at regular time intervals to keep the position delta-neutral (which is how gamma scalping makes money).
async def gamma_scalp(initial_interval=30, interval=120):
    await asyncio.sleep(initial_interval)
    maintain_delta_neutral()
    while True:
        await asyncio.sleep(interval)
        maintain_delta_neutral()


# THE MAIN EVENT #

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(
    trade_update_stream._run_forever(),
    #initial_trades(),
    gamma_scalp()
))
loop.close()




