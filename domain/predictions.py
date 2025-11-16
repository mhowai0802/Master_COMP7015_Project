from dataclasses import dataclass
from datetime import date
from typing import Optional

from .stocks import Stock


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



