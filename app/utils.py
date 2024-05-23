import pandas as pd

def normalizeTicker(ticker: str):
    return ticker.replace('.', '-')

def denormalizeTicker(ticker: str):
    return ticker.replace('-', '.')

def getSP500Tickers():
    tickerTable = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickerTable.loc[:, ["Symbol"]].head(500)
    return list(map(normalizeTicker, tickers["Symbol"].tolist()))
