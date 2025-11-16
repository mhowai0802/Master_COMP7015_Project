from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np

from domain.stocks import StockPriceSeries


def analyze_intraday_volatility(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze intraday volatility patterns by grouping data by 30-minute intervals.
    
    Parameters
    ----------
    intraday_df : pd.DataFrame
        DataFrame with datetime index and columns: open, high, low, close, volume
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hour, minute, time_slot, avg_volatility, avg_range, avg_volume, count
        Sorted by volatility descending.
    """
    if intraday_df.empty:
        return pd.DataFrame(columns=["hour", "minute", "time_slot", "avg_volatility", "avg_range", "avg_volume", "count"])
    
    # Extract hour and minute from datetime index
    intraday_df = intraday_df.copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(intraday_df.index, pd.DatetimeIndex):
        try:
            intraday_df.index = pd.to_datetime(intraday_df.index)
        except Exception:
            return pd.DataFrame(columns=["hour", "minute", "time_slot", "avg_volatility", "avg_range", "avg_volume", "count"])
    
    # Handle timezone-aware datetimes by converting to naive
    if intraday_df.index.tz is not None:
        intraday_df.index = intraday_df.index.tz_localize(None)
    
    # Extract hour and minute using pandas datetime accessor
    intraday_df["hour"] = intraday_df.index.hour
    intraday_df["minute"] = intraday_df.index.minute
    
    # Create 30-minute time slots (round down to nearest 30 minutes)
    # 0-29 minutes -> 0, 30-59 minutes -> 30
    intraday_df["minute_slot"] = (intraday_df["minute"] // 30) * 30
    
    # Create time_slot string for display (e.g., "9:30", "10:00")
    intraday_df["time_slot"] = intraday_df["hour"].astype(str) + ":" + intraday_df["minute_slot"].astype(str).str.zfill(2)
    
    # Calculate volatility metrics for each row
    # Volatility = (high - low) / close (normalized range)
    intraday_df["volatility"] = (intraday_df["high"] - intraday_df["low"]) / intraday_df["close"]
    intraday_df["range"] = intraday_df["high"] - intraday_df["low"]
    
    # Group by hour and minute_slot and aggregate
    time_stats = intraday_df.groupby(["hour", "minute_slot", "time_slot"]).agg({
        "volatility": ["mean", "std"],
        "range": "mean",
        "volume": "mean",
        "close": "count",  # count of observations
    }).reset_index()
    
    # Flatten column names
    time_stats.columns = ["hour", "minute", "time_slot", "avg_volatility", "std_volatility", "avg_range", "avg_volume", "count"]
    
    # Sort by average volatility descending
    time_stats = time_stats.sort_values("avg_volatility", ascending=False)
    
    return time_stats


def get_best_monitoring_hours(
    intraday_df: pd.DataFrame,
    daily_series: StockPriceSeries,
    predicted_direction: str,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Get the best hours to monitor based on predicted direction and historical patterns.
    
    Parameters
    ----------
    intraday_df : pd.DataFrame
        Intraday price data with datetime index
    daily_series : StockPriceSeries
        Daily price series to determine which days were positive/negative
    predicted_direction : str
        "up", "down", or "flat" - the predicted direction for today
    top_n : int
        Number of top hours to return (default: 3)
    
    Returns
    -------
    List[Dict]
        List of dicts with keys: hour, hour_label, avg_volatility, avg_range, avg_volume
        Sorted by volatility descending.
    """
    if intraday_df.empty:
        return []
    
    # Create a mapping of date -> daily return direction
    daily_directions = {}
    for i in range(1, len(daily_series.prices)):
        prev_close = daily_series.prices[i-1].close
        curr_close = daily_series.prices[i].close
        daily_date = daily_series.prices[i].date
        
        if curr_close > prev_close * 1.001:  # >0.1% increase
            direction = "up"
        elif curr_close < prev_close * 0.999:  # >0.1% decrease
            direction = "down"
        else:
            direction = "flat"
        
        daily_directions[daily_date] = direction
    
    # Filter intraday data to only days matching predicted direction
    intraday_filtered = intraday_df.copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(intraday_filtered.index, pd.DatetimeIndex):
        try:
            intraday_filtered.index = pd.to_datetime(intraday_filtered.index)
        except Exception:
            # If conversion fails, use all data without filtering
            intraday_filtered = intraday_df.copy()
            intraday_filtered["date"] = None
    
    # Handle timezone-aware datetimes
    if isinstance(intraday_filtered.index, pd.DatetimeIndex) and intraday_filtered.index.tz is not None:
        intraday_filtered.index = intraday_filtered.index.tz_localize(None)
    
    # Extract date from datetime index
    try:
        # Convert index to date - normalize first to remove time component
        dt_index = pd.to_datetime(intraday_filtered.index)
        intraday_filtered["date"] = dt_index.normalize().to_series().dt.date
    except Exception:
        # Fallback: use all data without date filtering
        intraday_filtered = intraday_df.copy()
        intraday_filtered["date"] = None
    
    # Match dates and filter
    matching_dates = [
        d for d, direction in daily_directions.items()
        if direction == predicted_direction
    ]
    
    if not matching_dates:
        # If no matching days, use all data
        matching_dates = list(daily_directions.keys())
    
    # Filter by matching dates if date column is available
    if intraday_filtered["date"].isna().all() or intraday_filtered["date"].isnull().all():
        # Date extraction failed, use all data
        intraday_filtered = intraday_df.copy()
    else:
        intraday_filtered = intraday_filtered[intraday_filtered["date"].isin(matching_dates)]
        if intraday_filtered.empty:
            # Fallback to all data
            intraday_filtered = intraday_df.copy()
    
    # Analyze volatility for filtered data
    time_stats = analyze_intraday_volatility(intraday_filtered)
    
    if time_stats.empty:
        return []
    
    # Convert to list of dicts
    results = []
    for _, row in time_stats.head(top_n).iterrows():
        hour = int(row["hour"])
        minute = int(row["minute"])
        time_slot = str(row["time_slot"])
        
        # Calculate end time (30 minutes later)
        end_hour = hour
        end_minute = minute + 30
        if end_minute >= 60:
            end_hour += 1
            end_minute = end_minute % 60
        
        # Format label (e.g., "9:30 AM - 10:00 AM")
        start_label = format_time_label(hour, minute)
        end_label = format_time_label(end_hour, end_minute)
        hour_label = f"{start_label} - {end_label}"
        
        results.append({
            "hour": hour,
            "minute": minute,
            "time_slot": time_slot,
            "hour_label": hour_label,
            "avg_volatility": float(row["avg_volatility"]),
            "avg_range": float(row["avg_range"]),
            "avg_volume": float(row["avg_volume"]),
            "count": int(row["count"]),
        })
    
    return results


def format_time_label(hour: int, minute: int = 0) -> str:
    """Format hour and minute as readable label (e.g., 14, 30 -> '2:30 PM')."""
    if hour < 12:
        if hour == 0:
            label = f"12:{minute:02d} AM"
        else:
            label = f"{hour}:{minute:02d} AM"
    elif hour == 12:
        label = f"12:{minute:02d} PM"
    else:
        label = f"{hour-12}:{minute:02d} PM"
    
    return label


def format_hour_label(hour: int, minute: int = 0) -> str:
    """Format hour and minute as readable label range (e.g., 14, 30 -> '2:30 PM - 3:00 PM')."""
    start_label = format_time_label(hour, minute)
    
    # Calculate end time (30 minutes later)
    end_hour = hour
    end_minute = minute + 30
    if end_minute >= 60:
        end_hour += 1
        end_minute = end_minute % 60
    
    end_label = format_time_label(end_hour, end_minute)
    
    return f"{start_label} - {end_label}"

