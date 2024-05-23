from fastapi_restful.tasks import repeat_every
from sqlmodel import Session
import yfinance as yf
from app import db_engine
from app.utils import denormalizeTicker, getSP500Tickers
from app.database import SP500DataTable

def getData(ticker: str):
    tickerObj = yf.Ticker(ticker)
    name = tickerObj.info['shortName']
    current_price = tickerObj.info['currentPrice']
    last_close_price = tickerObj.info['previousClose']
    percent_diff = round(
        number=((current_price - last_close_price) / abs(last_close_price)) * 100, 
        ndigits=2
    )
    return SP500DataTable(
        ticker=denormalizeTicker(ticker), 
        name=name, 
        current_price=current_price, 
        percent_diff=percent_diff
    )

@repeat_every(seconds=3600 * 1)
async def refresh_sp500_task():
    with Session(db_engine) as session:
        tickers = getSP500Tickers()
        for ticker in tickers:
            print(f'fetching data for {ticker}...')
            data = getData(ticker)
            session.add(data)

        session.commit()
