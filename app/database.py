from app import db_engine
from sqlmodel import SQLModel, Field

class ExampleTable(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)

class SP500DataTable(SQLModel, table=True):
    ticker: str = Field(primary_key=True)
    name: str = Field()
    current_price: float = Field()
    percent_diff: float = Field()

def create_tables():
    SQLModel.metadata.create_all(db_engine)
