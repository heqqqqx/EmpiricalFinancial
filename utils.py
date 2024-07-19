from datetime import datetime, timedelta
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

def display_first_last_rows(df, n=40):
    print(df.head(n))
    print(df.tail(n))

def count_observations(df):
    return len(df)

def deduce_periodicity(df):
    df.index = pd.to_datetime(df.index)
    diff = df.index.to_series().diff().dropna()
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

def handle_anomalies(df, column, stock_name, window=365):
    anomalies = detect_anomalies(df, column)
    plot_stock_data_with_anomalies(df, stock_name, column, anomalies, is_anomalies=True)
    print(f"Found {len(anomalies)} anomalies in {column}")
    df.loc[anomalies.index, column] = np.nan
    rolling_mean = df[column].rolling(window=window, center=True).mean()
    df.loc[:, column] = df[column].fillna(rolling_mean)
    return df

def plot_stock_data_with_anomalies(df, stock_name, column, anomalies, is_anomalies=True):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label='Data')
    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')

    legend = f'{stock_name} - {column} with Anomalies' if is_anomalies else f'{stock_name} - {column} after Anomalies Handling'
    plt.title(legend)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.show()

def calculate_moving_average(df, column, window):
    df[f'MA_{window}'] = df[column].rolling(window=window).mean()
    return df

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    df['daily_return'] = df['Close'].pct_change(fill_method=None)
    mean_return = df['daily_return'].mean()
    std_dev = df['daily_return'].std()
    return (mean_return - risk_free_rate) / std_dev

def plot_trends(df, ticker):
    plt.figure(figsize=(14, 7))
    df_filtered = df[df['Date'] >= datetime.now() - timedelta(days=3*365)]
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price')
    for column in df.columns:
        if column.startswith('MA_'):
            plt.plot(df_filtered['Date'], df_filtered[column], label=column)
    plt.title(f'Trend Analysis for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_stock_data_with_anomalies(data, stock_name, column, is_anomalies=True):
    '''
    Plot the stock data with anomalies
    :param data (dict): Dictionary containing the stock data
    :param stock_name (str): Stock name
    :param column (str): Column to plot
    :param is_anomalies (bool): Whether to plot anomalies or after handling anomalies
    :return: None, just plot the stock data
    '''
    df = data[stock_name]
    anomalies = detect_anomalies(df, column)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label='Data')
    plt.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')

    legend = f'{stock_name} - {column} with Anomalies' if is_anomalies else f'{stock_name} - {column} after Anomalies Handling'
    plt.title(legend)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.show()

def calculate_daily_return(df):
    """
    Calculate the daily return based on the opening and closing prices.
    :param df (DataFrame): Input DataFrame
    :return: DataFrame with 'Daily Return' column added
    """
    df['Daily Return'] = (df['Close'] - df['Open']) / df['Open']
    return df

def calculate_average_daily_return(df):
    """
    Calculate the average daily return for different periods.
    :param df (DataFrame): Input DataFrame
    :return: dict with average daily returns for different periods
    """
    df = calculate_daily_return(df)
    avg_daily_return = {
        'weekly': df['Daily Return'].resample('W').mean(),
        'monthly': df['Daily Return'].resample('ME').mean(),
        'yearly': df['Daily Return'].resample('YE').mean(),
    }
    return avg_daily_return

def calculate_average_prices(df):
    '''
    Calculate the average prices for different time periods
    :param df (dataFrame): Input DataFrame
    :return: dict, the average prices for different time periods
    '''

    avg_prices = {
        'daily': df[['Open', 'Close']].resample('D').mean(),
        'weekly': df[['Open', 'Close']].resample('W').mean(),
        'monthly': df[['Open', 'Close']].resample('ME').mean(),
        'yearly': df[['Open', 'Close']].resample('YE').mean(),
    }
    return avg_prices

def calculate_price_changes(df):
    """
    Calculate the price changes for a stock
    :param df (dataFrame): Input DataFrame
    :return: dataFrame, the DataFrame with the price changes added
    """
    df['Daily Change'] = df['Close'].diff()

    monthly_close = df['Close'].resample('ME').ffill()
    monthly_change = monthly_close.diff()
    monthly_change.name = 'Monthly Change'

    df = df.merge(monthly_change, left_index=True, right_index=True, how='left')
    df['Monthly Change'].fillna(method='ffill', inplace=True)

    return df

def calculate_bollinger_bands(df, window=20):
    """
    Calculate the Bollinger Bands for a stock
    :param df (dataFrame): Input DataFrame
    :param window (int): Window size for the moving average
    :return: dataFrame, the DataFrame with the Bollinger Bands added
    """
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['STD20'] = df['Close'].rolling(window=window).std()
    df['Upper Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower Band'] = df['MA20'] - (df['STD20'] * 2)
    return df

def plot_bollinger_bands(cleaned_data, tickers):
    """
    Plot the Bollinger Bands for different stocks
    :param cleaned_data (dict): Dictionary containing the cleaned stock data
    :param tickers (list): List of stock tickers
    :return: None, just plot the Bollinger Bands
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    for ticker in tickers:
        df = cleaned_data[ticker]


        df_filtered = df[df['Date'] >= start_date]
        df_filtered = calculate_bollinger_bands(df_filtered)

        plt.figure(figsize=(14, 7))
        plt.plot(df_filtered['Date'], df_filtered['Close'], label=f'{ticker} Close Price')
        plt.plot(df_filtered['Date'], df_filtered['Upper Band'], label='Upper Band', color='red')
        plt.plot(df_filtered['Date'], df_filtered['Lower Band'], label='Lower Band', color='green')
        plt.plot(df_filtered['Date'], df_filtered['MA20'], label='MA20',alpha=0.5)
        plt.fill_between(df_filtered['Date'], df_filtered['Lower Band'], df_filtered['Upper Band'], color='blue', alpha=0.1)

        plt.title(f'Bollinger Bands for {ticker} - Last 3 Years')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.show()

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a stock
    :param df (dataFrame): Input DataFrame
    :param short_window (int): Short window size for the EMA
    :param long_window (int): Long window size for the EMA
    :param signal_window (int): Signal line window size
    :return: dataFrame, the DataFrame with the MACD added
    """
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD Histogram'] = df['MACD'] - df['Signal Line']
    return df

def plot_macd(ticker):
    '''
    Plot the Moving Average Convergence Divergence (MACD) for a stock
    :param ticker (str): Stock ticker
    :return: None, just plot the MACD
    '''

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    df = cleaned_data[ticker]
    df = calculate_macd(df)
    df_filtered = df[df['Date'] >= start_date]

    plt.figure(figsize=(14, 7))
    plt.plot(df_filtered['Date'], df_filtered['MACD'], label='MACD', color='blue')
    plt.plot(df_filtered['Date'], df_filtered['Signal Line'], label='Signal Line', color='red')
    plt.bar(df_filtered['Date'], df_filtered['MACD Histogram'], label='MACD Histogram', color=['green' if val >= 0 else 'red' for val in df_filtered['MACD Histogram']], alpha=0.6)

    plt.title(f'MACD for {ticker} - Last 3 Years')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend(loc='best')
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.show()

def calculate_rsi(df, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a stock
    :param df (dataFrame): Input DataFrame
    :param window (int): Window size for the EMA
    :return: dataFrame, the DataFrame with the RSI added
    """
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    return df

def plot_close_price_with_rsi(df, ticker):
    """
    Plot the Close Price and Relative Strength Index (RSI) for a stock
    :param df (dataFrame): Input DataFrame
    :param ticker (str): Stock ticker
    :return: None, just plot the Close Price and RSI
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    plt.figure(figsize=(14, 10))

    #close price
    plt.subplot(2, 1, 1)
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price')
    plt.title(f'Close Price for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    #rsi 
    plt.subplot(2, 1, 2)
    plt.plot(df_filtered['Date'], df_filtered['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='purple', linestyle='--', linewidth=1)
    plt.axhline(30, color='purple', linestyle='--', linewidth=1)
    plt.fill_between(df_filtered['Date'], 30, 70, color='purple', alpha=0.1)
    plt.title(f'RSI for {ticker} - Last 3 Years')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend(loc='best')
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()

def plot_rsi(ticker):
    """
    Plot the Relative Strength Index (RSI) for a stock
    :param ticker (str): Stock ticker
    :return: None, just plot the RSI
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)


    df = cleaned_data[ticker]
    df = calculate_rsi(df)

    df_filtered = df[df['Date'] >= start_date]

    plt.figure(figsize=(14, 7))
    plt.plot(df_filtered['Date'], df_filtered['RSI'], label='RSI', color='purple')

    plt.axhline(70, color='purple', linestyle='--', linewidth=1)
    plt.axhline(30, color='purple', linestyle='--', linewidth=1)

    plt.fill_between(df_filtered['Date'], 30, 70, color='purple', alpha=0.1)

    plt.title(f'RSI for {ticker} - Last 3 Years')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend(loc='best')
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.show()

def plot_close_price_with_macd(df, ticker):
    '''
    Plot the Close Price and Moving Average Convergence Divergence (MACD) for a stock
    :param df (dataFrame): Input DataFrame
    :param ticker (str): Stock ticker
    :return: None, just plot the Close Price and MACD
    '''
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title(f'Close Price for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['Date'], df['Signal Line'], label='Signal Line', color='red')
    plt.bar(df['Date'], df['MACD Histogram'], label='MACD Histogram', color=['green' if val >= 0 else 'red' for val in df['MACD Histogram']], alpha=0.6)
    plt.title(f'MACD for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend(loc='best')
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

def calculate_ichimoku(df):
    '''
    Calculate the Ichimoku Cloud for a stock
    :param df (dataFrame): Input DataFrame
    :return: dataFrame, the DataFrame with the Ichimoku Cloud added
    '''
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Chikou_span'] = df['Close'].shift(-26)

    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    return df

def plot_closing_price_and_ichimoku(tickers, cleaned_data,):
    '''
    Plot the Closing Price and Ichimoku Cloud for different stocks
    :param tickers (list): List of stock tickers
    :param cleaned_data (dict): Dictionary containing the cleaned stock data
    :return: None, just plot the Closing Price and Ichimoku Cloud
    '''
    start_date = datetime.now() - timedelta(days=1*365)
    end_date = datetime.now()
    for ticker in tickers:
        df = cleaned_data[ticker]
        df = calculate_ichimoku(df)

        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        plt.figure(figsize=(14, 7))

        plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close', color='black')

        plt.plot(df_filtered['Date'], df_filtered['Tenkan_sen'], label='Tenkan-sen', color='red')
        plt.plot(df_filtered['Date'], df_filtered['Kijun_sen'], label='Kijun-sen', color='blue')
        plt.plot(df_filtered['Date'], df_filtered['Chikou_span'], label='Chikou Span', color='green')
        plt.plot(df_filtered['Date'], df_filtered['Senkou_span_a'], label='Senkou Span A', color='orange')
        plt.plot(df_filtered['Date'], df_filtered['Senkou_span_b'], label='Senkou Span B', color='purple')
        plt.fill_between(df_filtered['Date'], df_filtered['Senkou_span_a'], df_filtered['Senkou_span_b'],
                         where=df_filtered['Senkou_span_a'] >= df_filtered['Senkou_span_b'], color='green', alpha=0.2)
        plt.fill_between(df_filtered['Date'], df_filtered['Senkou_span_a'], df_filtered['Senkou_span_b'],
                         where=df_filtered['Senkou_span_a'] < df_filtered['Senkou_span_b'], color='red', alpha=0.2)

        plt.title(f'Ichimoku for {ticker} - Last Year')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True)

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.show()

def calculate_monthly_returns(df):
    '''
    Calculate the monthly returns for a stock
    :param df (dataFrame): Input DataFrame
    :return: dataFrame, the DataFrame with the monthly returns added
    '''
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Monthly Return'] = df['Close'].pct_change(periods=21, fill_method=None) * 100
    return df

def create_seasonality_table(df, start_year):
    """
    Create a seasonality table for a stock
    :param df (dataFrame): Input DataFrame
    :param start_year (int): Start year for the seasonality table
    :return: dataFrame, the seasonality table
    """
    df = df[df['Year'] >= start_year]
    seasonality_table = df.pivot_table(values='Monthly Return', index='Year', columns='Month', aggfunc='mean')
    seasonality_table.loc['Average'] = seasonality_table.mean()
    seasonality_table.loc['StDev'] = seasonality_table.std()
    seasonality_table.loc['Pos%'] = (seasonality_table > 0).mean() * 100
    return seasonality_table

def plot_seasonality_table(seasonality_table, ticker):
    """
    Plot the seasonality table for a stock
    :param seasonality_table (dataFrame): Seasonality table
    :param ticker (str): Stock ticker
    :return: None, just plot the seasonality table
    """
    plt.figure(figsize=(12, 8))

    cmap = sns.diverging_palette(30,130, as_cmap=True, s=300, l=50 )

    ax = sns.heatmap(seasonality_table, annot=True, fmt=".2f", cmap=cmap, center=0, cbar_kws={'label': 'Monthly Return (%)'})

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(month_names)

    ax.set_title(f'Seasonality Table for {ticker} - Last 10 Years')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    plt.show()

def process_and_plot_seasonality(start_year):
    """
    Process and plot the seasonality for different stocks
    :param start_year (int): Start year for the seasonality table
    :return: None, just plot the seasonality table
    """
    for ticker, df in data.items():
        df = calculate_monthly_returns(df)
        seasonality_table = create_seasonality_table(df, start_year)
        plot_seasonality_table(seasonality_table, ticker)
def ensure_datetime_index(df):
    """
    Ensure that the DataFrame has a datetime index.
    :param df (pd.DataFrame): Input DataFrame
    :return: pd.DataFrame with datetime index
    """
    if df.index.name != 'Date':
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
    return df

def calculate_indicators(df):
    df = ensure_datetime_index(df)

    # RSI
    df = calculate_rsi(df)

    # MACD
    df = calculate_macd(df)

    # Awesome Oscillator (AO)
    df = calculate_ao(df)

    # Ichimoku
    df = calculate_ichimoku(df)

    # Moving Averages
    df = calculate_moving_average(df, 'Close', 20)
    df = calculate_moving_average(df, 'Close', 50)

    # Bollinger Bands
    df = calculate_bollinger_bands(df)

    # Seasonality
    df = calculate_monthly_returns(df)

    return df

def calculate_ao(df, short_window=5, long_window=34):
    """
    Calculate the Awesome Oscillator (AO) for a stock
    :param df (dataFrame): Input DataFrame
    :param short_window (int): Short window size for the AO
    :param long_window (int): Long window size for the AO
    :return: dataFrame, the DataFrame with the AO added
    """
    median_price = (df['High'] + df['Low']) / 2
    df['AO'] = median_price.rolling(window=short_window).mean() - median_price.rolling(window=long_window).mean()
    return df

def plot_close_price_with_ao(df, ticker):
    '''
    Plot the Close Price and Awesome Oscillator (AO) for a stock
    :param df (dataFrame): Input DataFrame
    :param ticker (str): Stock ticker
    :return: None, just plot the Close Price and AO
    '''
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    df_filtered = df[df['Date'] >= start_date]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(df_filtered['Date'], df_filtered['Close'], label='Close Price')
    plt.title(f'Close Price for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    colors = ['green' if df_filtered['AO'].iloc[i] > df_filtered['AO'].iloc[i-1] else 'red' for i in range(1, len(df_filtered))]
    colors.insert(0, 'green')
    plt.subplot(2, 1, 2)
    plt.bar(df_filtered.index, df_filtered['AO'], label='AO', color=colors, alpha=0.6)
    plt.title(f'Awesome Oscillator for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('AO')
    plt.legend(loc='best')
    plt.grid(True)
    #
    plt.subplot(2, 1, 1)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
    #   
    plt.subplot(2, 1, 2)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(''))
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())

    plt.tight_layout()
    plt.show()

def simulate_portfolio(data, initial_balance=10000, risk_profile='modéré', strategy='ao',months=6):
    '''
    Simulate a portfolio based on a given risk profile and strategy, during a specific period of time
    :param data (dict): Dictionary containing the stock data
    :param initial_balance (float): Initial balance for the portfolio
    :param risk_profile (str): Risk profile for the portfolio
    :param strategy (str): Trading strategy to use
    :param months (int): Number of months to simulate
    :return: dict, the portfolio with the stock allocations
    '''
    allocations = {
        'défensif': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'modéré': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'agressif': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'everything': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15}
    }

    if risk_profile not in allocations:
        raise ValueError(f"Invalid risk profile: {risk_profile}. Choose from {list(allocations.keys())}")

    portfolio = {ticker: 0 for ticker in data.keys()}
    balance = initial_balance

    start_date = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    date_range = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')

    for ticker, df in data.items():
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except:
            pass

    for date in date_range:
        for ticker, df in data.items():
            try:
                if date in df.index:
                    allocation = allocations[risk_profile].get(ticker, 0)
                    amount_to_invest = balance * allocation

                    if strategy == 'ao':
                        ao = df.loc[date, 'AO']
                        if ao > 0 and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif ao < 0 and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0

                    elif strategy == 'rsi':
                        rsi = df.loc[date, 'RSI']
                        if rsi < 30 and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif rsi > 70 and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0

                    elif strategy == 'macd':
                        macd = df.loc[date, 'MACD']
                        signal_line = df.loc[date, 'Signal Line']
                        if macd > signal_line and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif macd < signal_line and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0

                    elif strategy == 'ichimoku':
                        tenkan_sen = df.loc[date, 'Tenkan_sen']
                        kijun_sen = df.loc[date, 'Kijun_sen']
                        if tenkan_sen > kijun_sen and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif tenkan_sen < kijun_sen and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0

                    elif strategy == 'ma':
                        ma_short = df.loc[date, 'MA_20']
                        ma_long = df.loc[date, 'MA_50']
                        if ma_short > ma_long and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif ma_short < ma_long and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0

                    elif strategy == 'bollinger':
                        close = df.loc[date, 'Close']
                        lower_band = df.loc[date, 'Lower Band']
                        upper_band = df.loc[date, 'Upper Band']
                        if close < lower_band and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif close > upper_band and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0
                    elif strategy == 'seasonality':
                        monthly_avg = df.loc[date, 'Monthly Return']
                        if monthly_avg > 0 and portfolio[ticker] == 0:
                            shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                            if shares_to_buy > 0:
                                balance -= shares_to_buy * df.loc[date, 'Close']
                                portfolio[ticker] += shares_to_buy
                        elif monthly_avg < 0 and portfolio[ticker] > 0:
                            shares_to_sell = portfolio[ticker]
                            if shares_to_sell > 0:
                                balance += shares_to_sell * df.loc[date, 'Close']
                                portfolio[ticker] = 0




            except KeyError as e:
                print(f"KeyError for {ticker} on {date}: {e}")
            except Exception as e:
                print(f"Exception for {ticker} on {date}: {e}")

    final_value = balance
    for ticker, shares in portfolio.items():
        if shares > 0:
            last_date = data[ticker].index[-1]
            final_value += shares * data[ticker].loc[last_date, 'Close']

    return final_value, portfolio


def simulate_combined_strategy(data, initial_balance=10000, risk_profile='everything', months=6):
    allocations = {
        'défensif': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'modéré': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'agressif': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15},
        'everything': {'AAPL': 0.15, 'MSFT': 0.15, 'AMZN': 0.15, 'GOOG': 0.15, 'TSLA': 0.15, 'ZM': 0.15, 'META': 0.15}
    }

    risk_parameters = {
        'défensif': {'buy_threshold': 2, 'sell_threshold': 1},
        'modéré': {'buy_threshold': 1, 'sell_threshold': 2},
        'agressif': {'buy_threshold': 1, 'sell_threshold': 3}
    }

    if risk_profile not in allocations or risk_profile not in risk_parameters:
        raise ValueError(f"Invalid risk profile: {risk_profile}. Choose from {list(allocations.keys())}")

    portfolio = {ticker: 0 for ticker in data.keys()}
    balance = initial_balance

    start_date = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    date_range = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d')

    total_buy_signals = 0
    total_sell_signals = 0

    buy_threshold = risk_parameters[risk_profile]['buy_threshold']
    sell_threshold = risk_parameters[risk_profile]['sell_threshold']
    print(f"Buy threshold: {buy_threshold}")
    print(f"Sell threshold: {sell_threshold}")
    for ticker, df in data.items():
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except:
            pass

    for date in date_range:
        for ticker, df in data.items():
            try:
                if date in df.index:
                    allocation = allocations[risk_profile].get(ticker, 0)
                    amount_to_invest = balance * allocation

                    ao = df.loc[date, 'AO']
                    rsi = df.loc[date, 'RSI']
                    macd = df.loc[date, 'MACD']
                    signal_line = df.loc[date, 'Signal Line']
                    tenkan_sen = df.loc[date, 'Tenkan_sen']
                    kijun_sen = df.loc[date, 'Kijun_sen']
                    ma_short = df.loc[date, 'MA_20']
                    ma_long = df.loc[date, 'MA_50']
                    close = df.loc[date, 'Close']
                    lower_band = df.loc[date, 'Lower Band']
                    upper_band = df.loc[date, 'Upper Band']
                    monthly_avg = df.loc[date, 'Monthly Return']
                    buy_signals = 0
                    sell_signals = 0

                    if ao > 0:
                        buy_signals += 1
                    if rsi < 30:
                        buy_signals += 1
                    if macd > signal_line:
                        buy_signals += 1
                    if tenkan_sen > kijun_sen:
                        buy_signals += 1
                    if ma_short > ma_long:
                        buy_signals += 1
                    if close < lower_band:
                        buy_signals += 1
                    if monthly_avg > 0:
                        buy_signals += 1

                    if ao < 0:
                        sell_signals += 1
                    if rsi > 70:
                        sell_signals += 1
                    if macd < signal_line:
                        sell_signals += 1
                    if tenkan_sen < kijun_sen:
                        sell_signals += 1
                    if ma_short < ma_long:
                        sell_signals += 1
                    if close > upper_band:
                        sell_signals += 1
                    if monthly_avg < 0:
                        sell_signals += 1

                    if buy_signals >= buy_threshold and portfolio[ticker] == 0:
                        shares_to_buy = int(amount_to_invest / df.loc[date, 'Close'])
                        if shares_to_buy > 0:
                            total_buy_signals += 1
                            balance -= shares_to_buy * df.loc[date, 'Close']
                            portfolio[ticker] += shares_to_buy
                            print(f"Achat de {shares_to_buy} stocks de {ticker} à {df.loc[date, 'Close']}$ le {date}")

                    elif sell_signals >= sell_threshold and portfolio[ticker] > 0:
                        shares_to_sell = portfolio[ticker]
                        if shares_to_sell > 0:
                            total_sell_signals += 1
                            balance += shares_to_sell * df.loc[date, 'Close']
                            portfolio[ticker] = 0
                            print(f"Vente de {shares_to_sell} stocks de {ticker} à {df.loc[date, 'Close']}$ le {date}")
            except KeyError as e:
                print(f"KeyError for {ticker} on {date}: {e}")
            except Exception as e:
                print(f"Exception for {ticker} on {date}: {e}")

    print(f"Total buy signals: {total_buy_signals}")
    print(f"Total sell signals: {total_sell_signals}")
    final_value = balance
    for ticker, shares in portfolio.items():
        if shares > 0:
            last_date = data[ticker].index[-1]
            final_value += shares * data[ticker].loc[last_date, 'Close']

    return final_value, portfolio

def calculate_daily_return(df):
    df['Daily Return'] = (df['Close'] - df['Open']) / df['Open']
    return df

def ensure_date_column(df):
    if df.index.name == 'Date':
        df = df.reset_index()
    return df