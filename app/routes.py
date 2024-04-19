from app import api

import pandas as pd
import yfinance as yf

@api.get('/')
def index():
    return "Hello World"

@api.get('/ping')
def ping():
    return "pong"

@api.get('/sp500')
def sp500():
    tickerTable = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    print(tickerTable.head())
    tickersAndNames = tickerTable.loc[:, ["Symbol", "Security"]]
    return tickersAndNames.to_dict(orient="records")
