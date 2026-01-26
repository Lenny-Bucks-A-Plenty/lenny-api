import io
import pandas as pd
import urllib.request


def normalizeTicker(ticker: str):
    return ticker.replace(".", "-")


def denormalizeTicker(ticker: str):
    return ticker.replace("-", ".")


def getSP500Tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    with urllib.request.urlopen(request) as response:
        html = response.read().decode("utf-8")
    tickerTable = pd.read_html(io.StringIO(html))[0]
    tickers = tickerTable.loc[:, ["Symbol"]].head(500)
    return list(map(normalizeTicker, tickers["Symbol"].tolist()))
