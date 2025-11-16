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


@dataclass
class PredictionInput:
    """
    Input to the prediction system.
    """

    stock: Stock
    as_of_date: date
    horizon_days: int  # how many days into the future we care about


@dataclass
class PredictionOutput:
    """
    Output from the prediction system: buy/sell levels and direction.
    """

    should_buy: bool
    should_sell: bool
    expected_direction: str  # "up", "down", or "flat"
    suggested_buy_price: Optional[float]
    suggested_sell_price: Optional[float]
    confidence: float  # 0–1


