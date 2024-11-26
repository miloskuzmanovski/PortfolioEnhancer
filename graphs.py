import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_historical_price(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label=f'{ticker} Closing Price')
    plt.title(f'{ticker} Historical Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'static/graphs/{ticker}_historical_price.png')
    plt.close()

def plot_moving_averages(data, ticker):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label=f'{ticker} Closing Price')
    plt.plot(data['Date'], data['MA50'], label='50-Day MA', linestyle='--')
    plt.plot(data['Date'], data['MA200'], label='200-Day MA', linestyle='--')
    plt.title(f'{ticker} Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'static/graphs/{ticker}_moving_averages.png')
    plt.close()

def plot_rsi(data, ticker):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], rsi, label=f'{ticker} RSI')
    plt.title(f'{ticker} Relative Strength Index (RSI)')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'static/graphs/{ticker}_rsi.png')
    plt.close()

def generate_graphs_for_tickers(tickers):
    for ticker in tickers:
        # Assume the CSV file is named after the ticker and placed in a specific directory
        csv_file = f"/sp500_data/{ticker}.csv"  # Update this path accordingly
        
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            data['Date'] = pd.to_datetime(data['Date'])

            plot_historical_price(data, ticker)
            plot_moving_averages(data, ticker)
            plot_rsi(data, ticker)
        else:
            print(f"CSV file for {ticker} not found at {csv_file}")

tickers = ["AAPL", "GOOGL", "MSFT"]
generate_graphs_for_tickers(tickers)

