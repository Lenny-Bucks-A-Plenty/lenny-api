import logging
import random
import time

from fastapi_restful.tasks import repeat_every
from sqlmodel import Session
from starlette.concurrency import run_in_threadpool
import yfinance as yf

from app import db_engine
from app.utils import denormalizeTicker, getSP500Tickers
from app.database import SP500DataTable

logger = logging.getLogger(__name__)


def getData(ticker: str):
    tickerObj = yf.Ticker(ticker)
    name = tickerObj.info["shortName"]
    current_price = tickerObj.info["currentPrice"]
    last_close_price = tickerObj.info["previousClose"]
    percent_diff = round(
        number=((current_price - last_close_price) / abs(last_close_price)) * 100,
        ndigits=2,
    )
    return SP500DataTable(
        ticker=denormalizeTicker(ticker),
        name=name,
        current_price=current_price,
        percent_diff=percent_diff,
    )


def refresh_sp500_once() -> None:
    logger.info("Starting SP500 data refresh task...")
    with Session(db_engine) as session:
        tickers = getSP500Tickers()
        logger.info("Refreshing SP500 data for %s tickers...", len(tickers))
        for ticker in tickers:
            logger.debug("Fetching data for %s...", ticker)
            data = getData(ticker)
            session.add(data)
            time.sleep(random.uniform(1,3))

        session.commit()
    logger.info("SP500 data refresh complete.")


@repeat_every(seconds=3600 * 1, logger=logger)
async def refresh_sp500_task() -> None:
    await run_in_threadpool(refresh_sp500_once)
