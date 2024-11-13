import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm


def load_trade_logs(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data


def filter_by_ticker(data, ticker):
    return data[data['Ticker'] == ticker]


def extract_price_levels(data):
    entry_prices = data['Entry Price'].dropna()
    exit_prices = data['Exit Price'].dropna()
    price_levels = pd.concat([entry_prices, exit_prices], axis=0)
    return price_levels

def identify_liquidity_zones(price_levels):
    liquidity_zones = price_levels.value_counts().sort_index(ascending=False)
    return liquidity_zones


def analyze_stop_take_levels(data):
    stop_losses = data['Stop Loss'].dropna()
    take_profits = data['Take Profit'].dropna()
    stop_take_levels = pd.concat([stop_losses, take_profits], axis=0)
    return stop_take_levels

#   High volume levels
def analyze_volume_levels(data):
    volume_data = data[['Volume', 'Exit Price']].dropna()
    high_volume_levels = volume_data.groupby('Exit Price').sum()
    return high_volume_levels

# Time series
def time_series_analysis(data):
    data['Rolling Avg Price'] = data['Price'].rolling(window=10).mean()
    data['Rolling Avg Volume'] = data['Volume'].rolling(window=10).mean()
    return data

# Barone-Adesi and Whaley for options
def baw_option_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000, option_type="call"):
    payoff_sum = 0
    for _ in range(num_simulations):
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.normal())
        payoff = max(0, ST - K) if option_type == "call" else max(0, K - ST)
        payoff_sum += payoff
    price = np.exp(-r * T) * (payoff_sum / num_simulations)
    return price

# Calculate option prices for each trade
def calculate_option_prices(data, model="baw"):
    data['Option Price'] = data.apply(
        lambda row: baw_option_price(row['Entry Price'], row['Strike Price'], row['Time to Expiry'],
                                     row['Risk-free Rate'], row['Volatility'], option_type=row['Option Type'])
        if model == "baw" else 
        monte_carlo_option_price(row['Entry Price'], row['Strike Price'], row['Time to Expiry'],
                                 row['Risk-free Rate'], row['Volatility'], option_type=row['Option Type']),
        axis=1
    )
    return data


def identify_supreme_levels(liquidity_zones, stop_take_levels, high_volume_levels, option_prices):
    supreme_levels = pd.merge(liquidity_zones, stop_take_levels, left_index=True, right_index=True, how='outer')
    supreme_levels = pd.merge(supreme_levels, high_volume_levels, left_index=True, right_index=True, how='outer')
    supreme_levels = pd.merge(supreme_levels, option_prices[['Option Price']], left_index=True, right_index=True, how='outer')
    supreme_levels.fillna(0, inplace=True)
    supreme_levels['Supreme Score'] = supreme_levels.sum(axis=1)  # Sum all factors, including option prices
    supreme_levels = supreme_levels.sort_values(by='Supreme Score', ascending=False)
    return supreme_levels


def analyze_trades(file_path, ticker, model="baw"):
    data = load_trade_logs(file_path)
    data = filter_by_ticker(data, ticker)
    
    price_levels = extract_price_levels(data)
    liquidity_zones = identify_liquidity_zones(price_levels)
    stop_take_levels = analyze_stop_take_levels(data)
    high_volume_levels = analyze_volume_levels(data)
    
    data = time_series_analysis(data)
    

    data = calculate_option_prices(data, model=model)


    supreme_levels = identify_supreme_levels(liquidity_zones, stop_take_levels, high_volume_levels, data)
    
    
    plot_zones(liquidity_zones, stop_take_levels, high_volume_levels, supreme_levels, title=f"{ticker} Trading Analysis")


file_path = input("Enter the CSV file path containing trade logs: ")
ticker = input("Enter the stock ticker symbol for analysis: ")
analyze_trades(file_path, ticker)
