import pandas as pd
import yfinance as yf
import os
from datetime import datetime

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    tables = pd.read_html(url)
    df = tables[0]

    # Get the list of tickers
    tickers = df['Symbol'].tolist()
    return tickers

tickers = get_sp500_tickers()
print(tickers[:5])

def download_sp500_data(tickers, start_date, end_date, folder='sp500_data'):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)
            filename = os.path.join(folder, f"{ticker}.csv")
            data.to_csv(filename)
            print(f"Data for {ticker} saved to {filename}")
        except Exception as e:
            print(f"Could not download data for {ticker}: {e}")

tickers = get_sp500_tickers()

start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

download_sp500_data(tickers, start_date, end_date)