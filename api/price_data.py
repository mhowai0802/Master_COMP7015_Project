from datetime import datetime, timedelta
import json
import os
from typing import Optional

import pandas as pd
import yfinance as yf

from domain.stocks import Stock, StockPriceSeries, PricePoint
from api.news_sentiment import fetch_news_sentiment
from config.api_keys import NEWSAPI_KEY


def _cache_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
    os.makedirs(base, exist_ok=True)
    return base


def _price_cache_path(stock: Stock, end_date: datetime, lookback_days: int) -> str:
    key = f"{stock.ticker}_{end_date.date().isoformat()}_{lookback_days}.json"
    return os.path.join(_cache_dir(), "prices_" + key)


def fetch_price_history(
    stock: Stock,
    end_date: Optional[datetime] = None,
    lookback_days: int = 365,
) -> StockPriceSeries:
    """
    Fetch historical daily price data for a stock using yfinance.
    Results are cached as JSON in a local folder to speed up repeated runs.
    """

    if end_date is None:
        end_date = datetime.today()

    cache_path = _price_cache_path(stock, end_date, lookback_days)

    # Try load from cache first
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        prices = [
            PricePoint(
                date=datetime.fromisoformat(p["date"]).date(),
                open=p["open"],
                high=p["high"],
                low=p["low"],
                close=p["close"],
                volume=p["volume"],
            )
            for p in raw.get("prices", [])
        ]
        return StockPriceSeries(stock=stock, prices=prices)

    # No cache → call yfinance and then cache
    start_date = end_date - timedelta(days=lookback_days)
    ticker = yf.Ticker(stock.ticker)
    hist = ticker.history(start=start_date, end=end_date)

    prices = []
    for idx, row in hist.iterrows():
        prices.append(
            PricePoint(
                date=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
            )
        )

    # Save to JSON cache
    payload = {
        "ticker": stock.ticker,
        "end_date": end_date.date().isoformat(),
        "lookback_days": lookback_days,
        "prices": [
            {
                "date": p.date.isoformat(),
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
            for p in prices
        ],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return StockPriceSeries(stock=stock, prices=prices)


def fetch_current_price(stock: Stock) -> Optional[float]:
    """
    Fetch the latest close price for a stock.
    """

    ticker = yf.Ticker(stock.ticker)
    data = ticker.history(period="1d")
    if data.empty:
        return None
    return float(data["Close"].iloc[-1])


# Sentiment and fundamental data.
def fetch_sentiment_score(stock: Stock, newsapi_key: Optional[str] = None) -> Optional[float]:
    """
    Sentiment-based signal using NewsAPI headlines.

    Parameters
    ----------
    stock : Stock
        Stock we want a sentiment score for.
    newsapi_key : str, optional
        Your NewsAPI API key. If not provided, this returns None.

    Returns
    -------
    Optional[float]
        Average sentiment score of recent news headlines in [-1, 1],
        where positive values indicate positive sentiment.
    """

    key = newsapi_key or NEWSAPI_KEY
    if not key:
        return None

    try:
        return fetch_news_sentiment(stock, api_key=key)
    except Exception:
        # In production you may want to log this; for now, fail gracefully.
        return None


def fetch_fundamental_snapshot(stock: Stock) -> dict:
    """
    Quick snapshot of fundamentals from Yahoo Finance.

    Returns a small dict with numeric features suitable for models and display:
    - market_cap
    - pe_ratio
    - ps_ratio
    - dividend_yield
    - profit_margin
    - revenue_growth
    """

    ticker = yf.Ticker(stock.ticker)
    info = {}
    try:
        # yfinance may expose both fast_info and info; use what we can.
        raw_info = ticker.info or {}
        fast = getattr(ticker, "fast_info", None)
        if fast:
            info["market_cap"] = float(getattr(fast, "market_cap", raw_info.get("marketCap", 0.0)) or 0.0)
        else:
            info["market_cap"] = float(raw_info.get("marketCap", 0.0) or 0.0)

        info["pe_ratio"] = float(raw_info.get("trailingPE", 0.0) or 0.0)
        info["ps_ratio"] = float(raw_info.get("priceToSalesTrailing12Months", 0.0) or 0.0)
        info["dividend_yield"] = float(raw_info.get("dividendYield", 0.0) or 0.0)
        info["profit_margin"] = float(raw_info.get("profitMargins", 0.0) or 0.0)
        info["revenue_growth"] = float(raw_info.get("revenueGrowth", 0.0) or 0.0)
    except Exception:
        # Fail gracefully if Yahoo changes fields or rate limits.
        info = {}

    return info


