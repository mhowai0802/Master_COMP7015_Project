from typing import List, Optional, Dict
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


def fetch_news_headlines(stock: Stock, api_key: str, page_size: int = 20) -> List[Dict[str, str]]:
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
        items = cached.get("headlines", [])
        # Backward compatibility: old cache stored plain strings
        normalized: List[Dict[str, str]] = []
        for it in items:
            if isinstance(it, str):
                normalized.append({"title": it, "publishedAt": ""})
            else:
                normalized.append(
                    {
                        "title": it.get("title", ""),
                        "publishedAt": it.get("publishedAt", ""),
                    }
                )
        return normalized

    resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    articles = data.get("articles", [])
    headlines: List[Dict[str, str]] = []
    for a in articles:
        title = a.get("title")
        published_at = a.get("publishedAt", "")
        if title:
            headlines.append({"title": title, "publishedAt": published_at})

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

    items = fetch_news_headlines(stock, api_key=api_key)
    titles = [it["title"] for it in items if it.get("title")]
    return compute_sentiment_from_headlines(titles)


