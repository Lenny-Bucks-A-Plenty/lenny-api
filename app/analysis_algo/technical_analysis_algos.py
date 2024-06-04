import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


"""
Work in progress
"""

class technical_analysis_algos:
    def bollinger_bands(ticker_df, period=20, std=2):
        data = ticker_df

        # Calculate the 20-period Simple Moving Average (SMA)
        data['simple_moving_ave'] = data['Close'].rolling(window=period).mean()

        # Calculate the 20-period Standard Deviation (SD)
        data['std'] = data['simple_moving_ave'].rolling(window=period).std()

        # Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB)
        data['upper_band'] = data['simple_moving_ave'] + 2 * data['std']
        data['lower_band'] = data['simple_moving_ave'] - 2 * data['std']

        data['difference'] = data['upper_band'] - data['lower_band']

        return data['difference'].iloc[-5:].mean()

    def momentum_oscillators(ticker_df):
        data = ticker_df.copy() 
        data['momentum'] = 0  
        
        for i in range(1, len(data)):
            data.loc[data.index[i], 'momentum'] = (data['Close'].iloc[i] / data['Close'].iloc[i-1]) * 100
            # values over 100 mean price is increasing, below 100 mean they are decreasing
        return data['momentum'].iloc[-5:].mean()

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

        # # Plot the actual vs predicted close prices
        # plt.plot(y_test.index, y_test, label='Actual Close Price')
        # plt.plot(y_test.index, y_pred, label='Predicted Close Price')
        # plt.legend()
        # plt.xlabel('Date')
        # plt.ylabel('Close Price')
        # plt.title('Actual vs Predicted Close Price')
        # plt.show()

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

        return future_df

    msft = yf.Ticker("MSFT")

    # get all stock info
    msft.info

    # get historical market data
    hist = msft.history(period="6mo", interval='1d')

    # boll_band = bollinger_bands(hist)

    # plt.plot(boll_band.index.values, boll_band['upper_band'])
    # plt.plot(boll_band.index.values, boll_band['lower_band'])
    # plt.plot(boll_band.index.values, boll_band['simple_moving_ave'])
    # plt.plot(boll_band.index.values, boll_band['Close'])
    # plt.show()

    # mo = momentum_oscillators(hist)

    # plt.plot(mo.index.values, mo['momentum'])
    # # plt.plot(mo.index.values, mo['Close'])
    # plt.show()

    # print(bollinger_bands(hist))
    # print(momentum_oscillators(hist))
    print(lin_reg(hist))
    # print(hist['Close'].iloc[-5:])

    """
    We need a way to value the bollinger, momentom, and linear regression out puts,
    I think, we should look at the last 7 days and look at the momentom and take the highest momentom 
    value for the past x days, look at the spread between the last n days for bollinger, and look at 
    the highest predicted percent increase over the past x days. These 3 factors can be used to
    determine which stock is the best pick to buy.

    make a voting system between the 3 stock analysis algorithms 
    """


