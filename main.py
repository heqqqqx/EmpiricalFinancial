import pandas as pd
import numpy as np
import yfinance as yf

# Define the ticker symbol and the period for data extraction
ticker_symbol = "TSLA"
start_date = "1997-05-15"  # Amazon's IPO date
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

# Fetch the historical data for AMZN
amzn_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Calculate daily returns
amzn_data['Daily Return'] = amzn_data['Adj Close'].pct_change()

# Calculate the average daily return
average_daily_return = amzn_data['Daily Return'].mean()

print(average_daily_return)