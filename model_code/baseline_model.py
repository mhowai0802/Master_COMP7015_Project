from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd

from domain.stocks import StockPriceSeries
from domain.predictions import PredictionInput, PredictionOutput


def _to_dataframe(series: StockPriceSeries) -> pd.DataFrame:
    """
    Convert StockPriceSeries to a pandas DataFrame for analysis.
    """

    records = [
        {
            "date": p.date,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        }
        for p in series.prices
    ]
    df = pd.DataFrame.from_records(records)
    # Ensure datetime index (Timestamps) for safe comparison
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df


def simple_trend_signal(df: pd.DataFrame, as_of: date) -> Tuple[str, float]:
    """
    Very simple technical indicator:

    - Compare short-term vs long-term moving averages of the close price.
    - Return direction ("up", "down", "flat") and a confidence score.
    """

    df = df.copy()
    # Ensure we compare Timestamps with Timestamps, not with datetime.date
    as_of_ts = pd.to_datetime(as_of)
    df = df[df.index <= as_of_ts]
    if len(df) < 30:
        # Not enough data, stay neutral
        return "flat", 0.3

    df["ma_short"] = df["close"].rolling(window=10).mean()
    df["ma_long"] = df["close"].rolling(window=30).mean()
    latest = df.iloc[-1]
    ma_short = latest["ma_short"]
    ma_long = latest["ma_long"]

    if pd.isna(ma_short) or pd.isna(ma_long):
        return "flat", 0.3

    diff = ma_short - ma_long
    # Normalize diff by price to get a rough relative strength
    rel = diff / latest["close"]

    if rel > 0.02:
        # Short MA significantly above long MA
        confidence = min(0.9, float(rel * 10))
        return "up", confidence
    if rel < -0.02:
        confidence = min(0.9, float(abs(rel) * 10))
        return "down", confidence

    return "flat", 0.4


def recommend_prices(
    df: pd.DataFrame,
    as_of: date,
) -> Tuple[float, float]:
    """
    Produce simple suggested buy/sell levels:

    - Buy slightly below the latest close.
    - Sell at a small profit target above the latest close.
    """

    as_of_ts = pd.to_datetime(as_of)
    df = df[df.index <= as_of_ts]
    latest = df.iloc[-1]
    close = float(latest["close"])

    suggested_buy = close * 0.98  # 2% below
    suggested_sell = close * 1.05  # 5% above

    return suggested_buy, suggested_sell


def run_baseline_model(
    series: StockPriceSeries,
    prediction_input: PredictionInput,
) -> PredictionOutput:
    """
    End-to-end baseline prediction pipeline.

    Later you can replace this with a more advanced model, e.g. based on
    Transformers or other sequence models from your course materials.
    """

    df = _to_dataframe(series)

    direction, confidence = simple_trend_signal(df, prediction_input.as_of_date)
    buy_price, sell_price = recommend_prices(df, prediction_input.as_of_date)

    should_buy = direction == "up"
    # Very simple rule: consider "sell" if trend is down
    should_sell = direction == "down"

    return PredictionOutput(
        should_buy=should_buy,
        should_sell=should_sell,
        expected_direction=direction,
        suggested_buy_price=buy_price,
        suggested_sell_price=sell_price,
        confidence=confidence,
    )


