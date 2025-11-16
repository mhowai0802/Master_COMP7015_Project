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


def _intraday_cache_path(stock: Stock, period_days: int, interval: str) -> str:
    key = f"{stock.ticker}_intraday_{period_days}d_{interval}.json"
    return os.path.join(_cache_dir(), "intraday_" + key)


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
    - fiscal_year_end (month name, e.g. "December")
    - financial_currency (e.g. "USD")
    - last_fiscal_year_end_date (ISO date of last fiscal year end)
    - most_recent_quarter_end (ISO date of most recent reported quarter)
    - financials_last_two_years: list of {year, total_revenue, net_income}
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

        # Meta information about the report
        info["fiscal_year_end"] = raw_info.get("fiscalYearEnd")  # e.g. "December"
        info["financial_currency"] = raw_info.get("financialCurrency")

        # These are usually UNIX timestamps in yfinance's info
        last_fy_ts = raw_info.get("lastFiscalYearEnd")
        if last_fy_ts:
            try:
                info["last_fiscal_year_end_date"] = pd.to_datetime(
                    last_fy_ts, unit="s"
                ).date().isoformat()
            except Exception:
                pass

        most_recent_q_ts = raw_info.get("mostRecentQuarter")
        if most_recent_q_ts:
            try:
                info["most_recent_quarter_end"] = pd.to_datetime(
                    most_recent_q_ts, unit="s"
                ).date().isoformat()
            except Exception:
                pass

        # Basic last-two-year financials (income statement)
        try:
            fin = ticker.financials  # DataFrame with index as line items, columns as period end dates
            financials_summary = []
            if isinstance(fin, pd.DataFrame) and not fin.empty:
                cols = list(fin.columns)
                # take last two years (most recent columns)
                cols_sorted = sorted(cols)
                for col in cols_sorted[-2:]:
                    year = (
                        str(col.date().year)
                        if hasattr(col, "year")
                        else str(col)
                    )
                    def _get_line(name_options):
                        for nm in name_options:
                            if nm in fin.index:
                                try:
                                    return float(fin.loc[nm, col] or 0.0)
                                except Exception:
                                    return 0.0
                        return 0.0

                    total_revenue = _get_line(
                        ["Total Revenue", "TotalRevenue", "TotalRevenueNet"]
                    )
                    net_income = _get_line(
                        ["Net Income", "NetIncome", "Net Income Applicable To Common Shares"]
                    )
                    financials_summary.append(
                        {
                            "year": year,
                            "total_revenue": total_revenue,
                            "net_income": net_income,
                        }
                    )
            if financials_summary:
                info["financials_last_two_years"] = financials_summary
        except Exception:
            # ignore financials errors
            pass
    except Exception:
        # Fail gracefully if Yahoo changes fields or rate limits.
        info = {}

    return info


def fetch_intraday_history(
    stock: Stock,
    period_days: int = 60,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch historical intraday (hourly) price data for a stock using yfinance.
    Results are cached as JSON in a local folder to speed up repeated runs.
    
    Parameters
    ----------
    stock : Stock
        Stock to fetch data for.
    period_days : int
        Number of days of history to fetch (default: 60).
    interval : str
        Data interval: "1h" for hourly, "30m" for 30-minute, etc. (default: "1h").
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, open, high, low, close, volume
        Index is datetime, timezone-aware.
    """
    cache_path = _intraday_cache_path(stock, period_days, interval)
    
    # Try load from cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # Check if cache matches requested parameters
            cached_interval = raw.get("interval", "")
            if cached_interval != interval:
                # Interval mismatch, fetch fresh data
                pass
            else:
                df_data = []
                for item in raw.get("data", []):
                    try:
                        df_data.append({
                            "datetime": pd.to_datetime(item["datetime"]),
                            "open": float(item["open"]),
                            "high": float(item["high"]),
                            "low": float(item["low"]),
                            "close": float(item["close"]),
                            "volume": float(item["volume"]),
                        })
                    except (KeyError, ValueError, TypeError):
                        # Skip invalid entries
                        continue
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df.set_index("datetime", inplace=True)
                    # Ensure index is DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    # Handle timezone-aware datetimes
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    if not df.empty:
                        return df
        except (json.JSONDecodeError, IOError, KeyError, ValueError):
            # If cache is corrupted, continue to fetch fresh data
            pass
    
    # No cache or cache failed → call yfinance
    ticker = yf.Ticker(stock.ticker)
    
    # yfinance period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    # For intraday data, yfinance has limitations:
    # - 30m/1h intervals: max 60 days (2mo)
    # - 15m intervals: max 60 days (2mo)
    # - 5m/1m intervals: max 7 days (5d)
    # So we cap at 60 days for 30m intervals
    if interval in ["30m", "1h", "15m"]:
        # Limit to 60 days for these intervals
        effective_period_days = min(period_days, 60)
        if effective_period_days <= 5:
            period_str = "5d"
        elif effective_period_days <= 30:
            period_str = "1mo"
        else:
            period_str = "2mo"
    else:
        # For shorter intervals, use shorter periods
        if period_days <= 5:
            period_str = "5d"
        elif period_days <= 30:
            period_str = "1mo"
        else:
            period_str = "2mo"  # Cap at 2mo for intraday
    
    try:
        hist = ticker.history(period=period_str, interval=interval)
        
        if hist.empty:
            # Try with a shorter period as fallback
            if period_str != "5d":
                try:
                    hist = ticker.history(period="5d", interval=interval)
                except Exception:
                    pass
        
        if hist.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        
        # Convert to our format
        df = pd.DataFrame({
            "open": hist["Open"].astype(float),
            "high": hist["High"].astype(float),
            "low": hist["Low"].astype(float),
            "close": hist["Close"].astype(float),
            "volume": hist.get("Volume", pd.Series(0.0, index=hist.index)).astype(float),
        })
        
        # Ensure index is DatetimeIndex and handle timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Filter to only market hours if needed (9:30am-4pm ET)
        # yfinance returns data in market timezone
        # Note: We're lenient here - if filtering removes too much data, we keep all
        if not df.empty and len(df) > 10:  # Only filter if we have enough data
            try:
                # Only filter if index is DatetimeIndex with time component
                if isinstance(df.index, pd.DatetimeIndex):
                    # Try filtering by time, but don't fail if it doesn't work
                    filtered_df = df.between_time("09:30", "16:00")
                    # Only use filtered data if we still have at least 50% of original data
                    if not filtered_df.empty and len(filtered_df) >= len(df) * 0.5:
                        df = filtered_df
            except Exception:
                # If filtering fails, continue with all data
                pass
        
        # Save to cache only if we have data
        if not df.empty:
            payload = {
                "ticker": stock.ticker,
                "period_days": period_days,
                "interval": interval,
                "data": [
                    {
                        "datetime": idx.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                    for idx, row in df.iterrows()
                ],
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        
        return df
    except Exception as e:
        # Log the error but return empty DataFrame
        import sys
        print(f"Error fetching intraday data for {stock.ticker}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


