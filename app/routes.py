from app import api, db_engine
from app.database import SP500DataTable
from app.analysis_algo.v2 import run_algo
import logging

from app.tasks.refresh_sp500 import refresh_sp500_once
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool
import yfinance as yf

logger = logging.getLogger(__name__)


@api.get("/")
def index():
    return "Hello World"


@api.get("/ping")
def ping():
    return "pong"


@api.get("/refresh-sp500")
async def refresh_sp500():
    logger.info("Manually refreshing SP500 data...")
    await run_in_threadpool(refresh_sp500_once)
    return "SP500 data refreshed"


@api.get("/sp500")
def sp500():
    with Session(db_engine) as session:
        statement = select(SP500DataTable)
        results = session.exec(statement).all()
        return results


@api.get("/take")
def getLennyTake(ticker: str):
    res = run_algo(ticker=ticker)
    return {"action": res[0], "explanation": res[1]}


@api.get("/graph")
def getStockGraph(ticker: str):
    yticker = yf.Ticker(ticker=ticker)
    df = yticker.history(interval="1d", period="2y")[["Close"]]

    res = []
    for idx, row in df.iterrows():
        res.append({"timestamp": str(idx), "price": row["Close"]})
    return res
