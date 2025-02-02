import yfinance as yf
from datetime import datetime

# Yahoo Finance API
def fetch_stock_data(ticker, start_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))["Close"]
        return stock_data
    except Exception:
        return None