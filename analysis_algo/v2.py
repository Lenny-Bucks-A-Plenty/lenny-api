import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
import numpy as np
import pandas as pd

def calculate_bollinger_bands(ticker_df: DataFrame, period: int = 20, std_multiplier: int = 2):
    data = ticker_df.copy()

    # Calculate the Simple Moving Average (SMA)
    data["simple_moving_avg"] = data["Close"].rolling(window=period).mean()

    # Calculate the Standard Deviation (STD) on closing prices
    data["std"] = data["Close"].rolling(window=period).std()

    # Calculate the Upper and Lower Bollinger Bands
    data["upper_band"] = data["simple_moving_avg"] + std_multiplier * data["std"]
    data["lower_band"] = data["simple_moving_avg"] - std_multiplier * data["std"]

    return data[["Close", "simple_moving_avg", "upper_band", "lower_band"]]


def determine_bollinger_signal(data: DataFrame):
    # Get the last row of the DataFrame
    latest_data = data.iloc[-1]

    # Extract the closing price and the Bollinger Bands
    close_price = latest_data["Close"]
    upper_band = latest_data["upper_band"]
    lower_band = latest_data["lower_band"]

    # Determine the signal
    if close_price > upper_band:
        return "Sell"
    elif close_price < lower_band:
        return "Buy"
    else:
        return "Wait"


def momentum_oscillators(ticker_df: DataFrame):
    data = ticker_df.copy()

    # Calculate momentum as the percentage change from the previous day
    data["momentum"] = (data["Close"] / data["Close"].shift(1) - 1) * 100

    return data[["Close", "momentum"]]


def determine_momentum_signal(data: DataFrame):
    latest_data = data.iloc[-1]
    momentum = latest_data["momentum"]

    # Determine the signal based on momentum
    if momentum > 0:
        return "Buy"
    elif momentum < 0:
        return "Sell"
    else:
        return "Wait"


# Define the linear regression function with moving average for future predictions
def lin_reg(ticker_df, n=60, ma_window=5):
    y = ticker_df['Close']
    X = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = X[:-n], X[-n:], y[:-n], y[-n:]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate the moving average of the close prices
    ticker_df['Moving_Avg'] = ticker_df['Close'].rolling(window=ma_window).mean()

    # Predict the next 5 days using the moving average
    future_predictions = []
    last_known_data = ticker_df.iloc[-1].copy()  

    for _ in range(5):
        ma_close = ticker_df['Moving_Avg'].iloc[-ma_window:].mean()
        future_predictions.append(ma_close)

        # Simulate the next day's input data
        next_day_data = pd.DataFrame({
            'Open': [ma_close],
            'High': [ma_close],
            'Low': [ma_close],
            'Close': [ma_close],
            'Volume': [last_known_data['Volume']]  # Keep the volume the same as the previous day (or modify as needed)
        }, index=[ticker_df.index[-1] + pd.Timedelta(days=1)])

        # Append the simulated data to the DataFrame to maintain continuity
        ticker_df = pd.concat([ticker_df, next_day_data])
        ticker_df['Moving_Avg'] = ticker_df['Close'].rolling(window=ma_window).mean()
        last_known_data = ticker_df.iloc[-1]

    # Create a DataFrame for the future predictions
    future_dates = pd.date_range(start=ticker_df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted Close Price'])

    return future_df.values


def determine_linear_reg_signal(predictions: list):
    if predictions[-1] > predictions[0]:
        return "Buy"
    else:
        return "Sell"


def main():
    msft = yf.Ticker("MSFT")

    # Get historical market data
    hist = msft.history(period="6mo", interval="1d")

    # Calculate Bollinger Bands
    bollinger_data = calculate_bollinger_bands(hist)
    bollinger_signal = determine_bollinger_signal(bollinger_data)

    # Calculate Momentum Oscillators
    momentum_data = momentum_oscillators(hist)
    momentum_signal = determine_momentum_signal(momentum_data)

    # Linear Regression Prediction
    lin_reg_pred = lin_reg(hist)
    lin_reg_signal = determine_linear_reg_signal(lin_reg_pred)

    # Print signals and data
    print(f"Bollinger Bands Signal: {bollinger_signal}")
    print(f"Momentum Signal: {momentum_signal}")
    print(f"Linear Regression Signal: {lin_reg_signal}")
    print(f"Bollinger Bands Data:\n{bollinger_data.tail()}")
    print(f"Momentum Data:\n{momentum_data.tail()}")

    # Determine overall recommendation based on all signals
    signals = [bollinger_signal, momentum_signal, lin_reg_signal]
    buy_count = signals.count("Buy")
    sell_count = signals.count("Sell")

    if buy_count > sell_count:
        recommendation = "Buy"
    elif sell_count > buy_count:
        recommendation = "Sell"
    else:
        recommendation = "Wait"

    print(f"\nOverall Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
