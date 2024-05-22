import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas import DataFrame


def calculate_bollinger_bands(
    ticker_df: DataFrame, period: int = 20, std_multiplier: int = 2
):
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


def lin_reg(ticker_df: DataFrame, n: int = 60):
    y = ticker_df["Close"]
    X = ticker_df[["Open", "High", "Low", "Close", "Volume"]]

    # Training on all but the last 'n' values, testing on the last 'n' values
    X_train, X_test, y_train, y_test = X[:-n], X[-n:], y[:-n], y[-n:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plotting actual vs predicted close prices
    # plt.figure(figsize=(10, 6))
    # plt.plot(y_test.index, y_test, label="Actual Close Price")
    # plt.plot(y_test.index, y_pred, label="Predicted Close Price", linestyle="--")
    # plt.legend()
    # plt.xlabel("Date")
    # plt.ylabel("Close Price")
    # plt.title("Actual vs Predicted Close Price")
    # plt.show()

    return y_pred


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
