# Lenny API

## Setup

```bash
uv venv
uv lock
uv sync --dev
```

Create a `.env` file with at least:

```
DATABASE_URL=sqlite:///lenny.db
ENV=dev
```

## Run locally

```bash
uv run uvicorn app:api --reload
```
