from app import db
from sqlmodel import SQLModel, Field

class ExampleTable(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)

def create_tables():
    SQLModel.metadata.create_all(db)
