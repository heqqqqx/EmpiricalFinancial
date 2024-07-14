import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_csv_files(file_paths):
    data = {}
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        data[key] = df
    return data
def display_first_last_rows(df, n=40):
    print(df.head(n))
    print(df.tail(n))

def count_observations(df):
    return len(df)

def deduce_periodicity(df):
    df['Date'] = pd.to_datetime(df['Date'])
    diff = df['Date'].diff().dropna()
    return diff.mode()[0]

def descriptive_statistics(df):
    return df.describe()

def identify_missing_values(df):
    return df.isna().sum()

def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return df.fillna(df[numeric_columns].mean())

def detect_anomalies(df, column, window=365):
    rolling_df = df[column].rolling(window=window, center=True)
    Q1 = rolling_df.quantile(0.25)
    Q3 = rolling_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies

def handle_anomalies(df, column, window=365):
    print("test")
    anomalies = detect_anomalies(df, column, window)
    print(f"Found {len(anomalies)} anomalies in {column}")
    print(anomalies[['Date', column]])
    df.loc[anomalies.index, column] = np.nan
    return handle_missing_values(df)

def calculate_moving_average(df, column, window):
    df[f'MA_{window}'] = df[column].rolling(window=window).mean()
    return df

def calculate_correlation(cleaned_data):
    returns = {ticker: df['Close'].pct_change() for ticker, df in cleaned_data.items()}
    return pd.DataFrame(returns).corr()

def calculate_performance(df):
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    return (final_price - initial_price) / initial_price

def calculate_annualized_return(df):
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    n_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    return (final_price / initial_price) ** (1 / n_years) - 1

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    df['daily_return'] = df['Close'].pct_change()
    mean_return = df['daily_return'].mean()
    std_dev = df['daily_return'].std()
    return (mean_return - risk_free_rate) / std_dev

def calculate_volatility(df):
    df['daily_return'] = df['Close'].pct_change()
    return df['daily_return'].std()

def plot_trends(df, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    for column in df.columns:
        if column.startswith('MA_'):
            plt.plot(df['Date'], df[column], label=column)
    plt.title(f'Trend Analysis for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_normalized_prices(cleaned_data, tickers):
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        df = cleaned_data[ticker]
        df['normalized_close'] = df['Close'] / df['Close'].iloc[0]
        plt.plot(df['Date'], df['normalized_close'], label=ticker)
    plt.title('Normalized Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()
