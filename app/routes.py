from app import api, db_engine
from app.database import SP500DataTable
from app.analysis_algo.v2 import do_shit
from sqlmodel import Session, select

@api.get('/')
def index():
    return "Hello World"

@api.get('/ping')
def ping():
    return "pong"

@api.get('/sp500')
def sp500():
    with Session(db_engine) as session:
        statement = select(SP500DataTable)
        results = session.exec(statement).all()
        return results
    
@api.get('/take')
def getLennyTake(ticker: str):
    res = do_shit(ticker=ticker)
    return res
