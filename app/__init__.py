from fastapi import FastAPI
from sqlmodel import create_engine
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

#? ENVIRONMENT VARIABLES
load_dotenv(".env")
isDev = True if os.environ.get("ENV") == "dev" else False
DATABASE_URL = os.environ.get("DATABASE_URL")

#? DATABASE SETUP
db = create_engine(
    url = DATABASE_URL,
    echo = True,
    connect_args = { "check_same_thread": False }
)

from app import database

@asynccontextmanager
async def onStartup(app: FastAPI):
    database.create_tables()
    yield

#? API SETUP
api = FastAPI(
    debug = isDev,
    title = "Lenny Bucks A Plenty API",
    version = "0.1.0",
    lifespan = onStartup
)

from app import routes
