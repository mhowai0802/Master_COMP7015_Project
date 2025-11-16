from typing import List, Optional
import json
import os

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from domain.stocks import Stock


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


def _cache_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
    os.makedirs(base, exist_ok=True)
    return base


def _news_cache_path(stock: Stock) -> str:
    key = f"{stock.ticker}.json"
    return os.path.join(_cache_dir(), "news_" + key)


def fetch_news_headlines(stock: Stock, api_key: str, page_size: int = 20) -> List[str]:
    """
    Fetch recent news headlines about a given stock using NewsAPI.

    Parameters
    ----------
    stock : Stock
        Stock for which we want related news.
    api_key : str
        Your NewsAPI API key.
    page_size : int
        Number of news articles to fetch (max 100 per NewsAPI docs).
    """

    params = {
        "q": stock.ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    cache_path = _news_cache_path(stock)

    # Try cache first
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("headlines", [])

    resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", [])
    headlines: List[str] = []
    for a in articles:
        title = a.get("title")
        if title:
            headlines.append(title)

    # Save cache
    payload = {
        "ticker": stock.ticker,
        "headlines": headlines,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return headlines


def compute_sentiment_from_headlines(headlines: List[str]) -> Optional[float]:
    """
    Compute an aggregate sentiment score from a list of headlines
    using VADER sentiment analysis.

    Returns a score in approximately [-1, 1], where:
    -1 = very negative, 0 = neutral, 1 = very positive.
    """

    if not headlines:
        return None

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for h in headlines:
        vs = analyzer.polarity_scores(h)
        scores.append(vs["compound"])

    if not scores:
        return None

    return float(sum(scores) / len(scores))


def fetch_news_sentiment(stock: Stock, api_key: str) -> Optional[float]:
    """
    High-level helper: from stock -> news headlines -> sentiment score.
    """

    headlines = fetch_news_headlines(stock, api_key=api_key)
    return compute_sentiment_from_headlines(headlines)


