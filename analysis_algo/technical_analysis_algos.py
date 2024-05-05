import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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
        return data

    def momentum_oscillators(ticker_df):
        data = ticker_df.copy() 
        data['momentum'] = 0  
        
        for i in range(1, len(data)):
            data.loc[data.index[i], 'momentum'] = (data['Close'].iloc[i] / data['Close'].iloc[i-1]) * 100
            # values over 100 mean price is increasing, below 100 mean they are decreasing
        
        return data

    def lin_reg(ticker_df, n=60):
        y = ticker_df['Close']
        X = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # for training we get all but the last values, for testing we use the last values
        X_train, X_test, y_train, y_test = X[:-n], X[-n:], y[:-n], y[-n:] 
        
        model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        plt.plot(y_test.index, y_test, label='Actual Close Price', )
        plt.plot(y_test.index, y_pred, label='Predicted Close Price')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Actual vs Predicted Close Price')
        plt.show()

        return y_pred

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
    # plt.plot(mo.index.values, mo['Close'])
    # plt.show()

    log = lin_reg(hist)


