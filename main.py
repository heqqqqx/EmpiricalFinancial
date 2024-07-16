def calculate_indicators(df):
    """
    Calculate the technical indicators for a stock
    :param df (pd.DataFrame): Input DataFrame
    :return: pd.DataFrame with the technical indicators
    """
    df = ensure_datetime_index(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_ao(df)
    df = calculate_ichimoku(df)
    df = calculate_moving_average(df, 'Close', 20)
    df = calculate_moving_average(df, 'Close', 50)
    df = calculate_bollinger_bands(df)
    df = calculate_monthly_returns(df)

    return df