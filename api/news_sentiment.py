from typing import List, Optional, Dict
import json
import os

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from domain.stocks import Stock

# Import deep learning sentiment utilities
try:
    from ml.sentiment_utils import predict_sentiment_score
    DL_SENTIMENT_AVAILABLE = True
except ImportError:
    DL_SENTIMENT_AVAILABLE = False
    print("Warning: Deep learning sentiment models not available. Using VADER fallback.")


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


def compute_sentiment_from_headlines(
    headlines: List[str],
    use_dl: bool = True,
    model_type: str = "lstm",
) -> Optional[float]:
    """
    Compute an aggregate sentiment score from a list of headlines.
    
    Uses deep learning models (LSTM/BERT) if available, otherwise falls back to VADER.

    Parameters
    ----------
    headlines : List[str]
        List of headline strings.
    use_dl : bool
        Whether to use deep learning models (default: True).
    model_type : str
        Model type: "lstm" or "bert" (default: "lstm").

    Returns
    -------
    Optional[float]
        Average sentiment score in [-1, 1], where:
        -1 = very negative, 0 = neutral, 1 = very positive.
    """
    if not headlines:
        return None

    # Try deep learning models first
    if use_dl and DL_SENTIMENT_AVAILABLE:
        try:
            scores = predict_sentiment_score(
                headlines,
                model_type=model_type,
            )
            if isinstance(scores, list):
                return float(sum(scores) / len(scores))
            return float(scores)
        except Exception as e:
            print(f"Warning: Deep learning sentiment failed: {e}. Falling back to VADER.")
    
    # Fallback to VADER
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


