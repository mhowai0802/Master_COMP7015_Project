from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass
class Stock:
    """
    Domain object representing a stock from your watchlist.
    """

    name: str
    ticker: str
    description: Optional[str] = None
    market_cap: Optional[float] = None  # in USD


@dataclass
class PricePoint:
    """
    Single OHLCV price point for a stock.
    """

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StockPriceSeries:
    """
    Time series of price points for a stock.
    """

    stock: Stock
    prices: List[PricePoint]


