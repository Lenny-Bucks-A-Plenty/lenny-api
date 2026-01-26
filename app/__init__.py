import asyncio
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import create_engine
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

# ? ENVIRONMENT VARIABLES
load_dotenv(".env")


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_configure_logging()
isDev = True if os.environ.get("ENV") == "dev" else False
DATABASE_URL = os.environ.get("DATABASE_URL")

# ? DATABASE SETUP
db_engine = create_engine(
    url=DATABASE_URL, echo=True, connect_args={"check_same_thread": False}
)

from app import database
from app.tasks.refresh_sp500 import refresh_sp500_task


@asynccontextmanager
async def onStartup(app: FastAPI):
    database.create_tables()
    # asyncio.create_task(refresh_sp500_task())
    yield


# ? API SETUP
api = FastAPI(
    debug=isDev, title="Lenny Bucks A Plenty API", version="0.1.0", lifespan=onStartup
)

api.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"])

from app import routes
