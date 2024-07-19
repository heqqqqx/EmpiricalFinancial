import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

from utils import display_first_last_rows, count_observations, deduce_periodicity, descriptive_statistics, identify_missing_values, handle_missing_values, handle_anomalies, plot_stock_data_with_anomalies, calculate_average_prices, calculate_price_changes, ensure_date_column, calculate_daily_return, calculate_moving_average, plot_trends, calculate_volatility, calculate_performance, calculate_annualized_return, calculate_sharpe_ratio, calculate_correlation, plot_bollinger_bands, calculate_macd, calculate_rsi, plot_close_price_with_rsi, plot_macd, plot_rsi, plot_close_price_with_macd, plot_closing_price_and_ichimoku, process_and_plot_seasonality, calculate_ao, plot_close_price_with_ao, calculate_indicators, simulate_portfolio, simulate_combined_strategy
tickers = ["AAPL", "MSFT","AMZN", "META", "GOOG", "TSLA", "ZM", ]
start_date = (datetime.now() - timedelta(days=30*365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
os.makedirs("data", exist_ok=True)

data = {}
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    data[ticker] = df
    df.to_csv(f"data/{ticker}_historical_data.csv")
    print(df.head())
    print(df.tail())


cleaned_data = {}

for ticker, df in data.items():
    print(f"\nExploration des données pour {ticker} :")
    print(f"Premières et dernières lignes :")
    display_first_last_rows(df)
for ticker, df in data.items():
    print(f"Nombre d'observations pour {ticker}: {count_observations(df)}")
for ticker, df in data.items():
    print(f"Période entre les points de données : {deduce_periodicity(df)}")

for ticker, df in data.items():
    print("Statistiques descriptives :")
    print(descriptive_statistics(df))
for ticker, df in data.items():
    print(f"Valeurs manquantes pour {ticker} :")
    print(identify_missing_values(df))
    df = handle_missing_values(df)  # Handle missing values
    cleaned_data[ticker] = df

for ticker, df in cleaned_data.items():
    print(f"Valeurs manquantes pour {ticker} après nettoyage :")
    print(identify_missing_values(df))


for ticker, df in data.items():
    print(f"Nettoyage des données pour {ticker} :")
    for column in ['Open', 'High', 'Low', 'Close']:
        print(f"Traitement de la colonne {column} :")
        df = handle_anomalies(df=df, column=column, stock_name=ticker)
    cleaned_data[ticker] = df

for ticker, df in cleaned_data.items():
    for column in ['Open', 'High', 'Low', 'Close']:
        plot_stock_data_with_anomalies(cleaned_data, ticker, column, is_anomalies=False)


os.makedirs("cleaned_data", exist_ok=True)
for ticker, df in cleaned_data.items():
    df.to_csv(f"cleaned_data/{ticker}_cleaned_data.csv", index=False)
    print(df.head())
    print(df.tail())

for ticker, df in cleaned_data.items():
    avg_prices = calculate_average_prices(df)
    print(f"\nAverage prices for {ticker}:")
    for period, prices in avg_prices.items():
        print(f"\n{period.capitalize()} average prices:")
        print(prices)
    print(df.head())
    print(df.tail())


for ticker, df in cleaned_data.items():
    df = calculate_price_changes(df)
    cleaned_data[ticker] = df
    print(f"\nPrice changes for {ticker}:")
    print(df[['Daily Change', 'Monthly Change']].fillna(df['Monthly Change'].mean()).head())
    print(df.head(30))
    print(df.tail(30))

for ticker, df in cleaned_data.items():
    df = ensure_date_column(df)  # S'assurer que 'Date' est dans les colonnes
    df = calculate_daily_return(df)
    cleaned_data[ticker] = df
    print(f"\nDaily return for {ticker}:")
    print(df[['Date', 'Daily Return']].dropna().head())
    print(df.head())
    print(df.tail())


average_daily_return = {}

for ticker, df in cleaned_data.items():
    df['Daily Return'] = df['Close'].pct_change(fill_method=None)
    avg_return = df['Daily Return'].mean()
    average_daily_return[ticker] = avg_return

best_stock = max(average_daily_return, key=average_daily_return.get)
print(f"\nStock with the highest average daily return: {best_stock} with an average return of {average_daily_return[best_stock]}")


def ensure_date_column(df):
    if df.index.name == 'Date':
        df = df.reset_index()
    return df

# Calculer le retour quotidien pour chaque action
for ticker, df in cleaned_data.items():
    df = ensure_date_column(df)
    df = calculate_daily_return(df)
    cleaned_data[ticker] = df
    print(f"\nDaily return for {ticker}:")
    print(df[['Date', 'Daily Return']].head())

for ticker, df in cleaned_data.items():
    df = calculate_moving_average(df, 'Close', 20)
    df = calculate_moving_average(df, 'Close', 50)
    df = calculate_moving_average(df, 'Close', 200)
    plot_trends(df, ticker)

for ticker, df in cleaned_data.items():
    print(f"\nStatistiques descriptives pour {ticker} :")
    print(descriptive_statistics(df))

for ticker, df in cleaned_data.items():
    print(f"Volatilité pour {ticker} : {calculate_volatility(df):.4f}")


for ticker, df in cleaned_data.items():
    print(df.head())
    print(df.tail())

performance_data = []
for ticker, df in cleaned_data.items():
    total_return = calculate_performance(df)
    annualized_return = calculate_annualized_return(df)
    sharpe_ratio = calculate_sharpe_ratio(df)
    volatility = calculate_volatility(df)
    performance_data.append({
        'Ticker': ticker,
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Volatility': volatility
    })

performance_df = pd.DataFrame(performance_data)
print(performance_df)

# Analyse de corrélation entre les actions
correlation_matrix = calculate_correlation(cleaned_data)
print("Matrice de corrélation :")
print(correlation_matrix)


plot_bollinger_bands(cleaned_data, tickers)


end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

for ticker in tickers:
    df = cleaned_data[ticker]
    df = calculate_macd(df)
    df = calculate_rsi(df)

    # Filter data for the last 3 years
    df_filtered = df[df['Date'] >= start_date]
    plot_close_price_with_rsi(df_filtered, ticker)
    plot_macd(ticker)

end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)
for ticker in tickers:
    df = cleaned_data[ticker]
    df = calculate_rsi(df)
    plot_rsi(ticker)


end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)
for ticker in tickers:
    df = cleaned_data[ticker]
    df = calculate_macd(df)

    # Filter data for the last 3 years
    df_filtered = df[df['Date'] >= start_date]
    plot_close_price_with_macd(df_filtered, ticker)



plot_closing_price_and_ichimoku(tickers, cleaned_data)

process_and_plot_seasonality(2014)

for ticker, df in cleaned_data.items():
    df = calculate_ao(df)
    cleaned_data[ticker] = df
    plot_close_price_with_ao(df, ticker)

for ticker, df in cleaned_data.items():
    cleaned_data[ticker] = calculate_indicators(df)


strategies = ['ao', 'rsi', 'macd', 'ichimoku', 'ma', 'bollinger','seasonality']
results = []

for strategy in strategies:
    final_value, portfolio = simulate_portfolio(cleaned_data, risk_profile='modéré', strategy=strategy)
    results.append({
        'Strategy': strategy,
        'Final Value': final_value,
        'Portfolio': portfolio
    })

for result in results:
    print(f"Strategy: {result['Strategy']}")
    print(f"Final Value: {result['Final Value']}")
    print(f"Portfolio: {result['Portfolio']}")
    print("\n")

risk_parameters = {
    'défensif': {'buy_threshold': 2, 'sell_threshold': 1},
    'modéré': {'buy_threshold': 1, 'sell_threshold': 2},
    'agressif': {'buy_threshold': 1, 'sell_threshold': 3}
}


# Exemple d'utilisation
profil = 'défensif'
duration = 6
final_value, portfolio = simulate_combined_strategy(cleaned_data, initial_balance=10000, risk_profile=profil, months=duration)
print(f"Profil risque : {profil}")
print(f"Durée : {duration} mois")
print(f"Valeur finale du portefeuille : {final_value}")
print(f"Composition finale du portefeuille : {portfolio}")