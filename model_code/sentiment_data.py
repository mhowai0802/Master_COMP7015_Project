"""
Data collection and preparation for sentiment analysis.

Collects news headlines from cache and downloads/prepares Financial PhraseBank dataset.
Creates train/val/test splits for model training.
"""

import json
import os
from typing import List, Tuple, Dict, Optional
from collections import Counter
import re

import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from domain.stocks import Stock


def _cache_dir() -> str:
    """Get cache directory path."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
    os.makedirs(base, exist_ok=True)
    return base


def _data_dir() -> str:
    """Get data directory for storing datasets."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(base, exist_ok=True)
    return base


def collect_news_headlines_from_cache() -> List[Dict[str, str]]:
    """
    Collect all news headlines from cache files.
    
    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries with 'text' and 'ticker' keys.
    """
    cache_dir = _cache_dir()
    headlines = []
    
    for filename in os.listdir(cache_dir):
        if filename.startswith("news_") and filename.endswith(".json"):
            cache_path = os.path.join(cache_dir, filename)
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                ticker = data.get("ticker", "")
                items = data.get("headlines", [])
                
                for item in items:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                    else:
                        title = str(item)
                    
                    if title:
                        headlines.append({
                            "text": title,
                            "ticker": ticker,
                        })
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue
    
    return headlines


def download_financial_phrasebank() -> Optional[pd.DataFrame]:
    """
    Download Financial PhraseBank dataset.
    
    Financial PhraseBank is a dataset of financial news sentences labeled with sentiment.
    We'll use a simplified approach: download from a public source or use a similar dataset.
    
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with 'text' and 'label' columns, or None if download fails.
    """
    data_dir = _data_dir()
    phrasebank_path = os.path.join(data_dir, "financial_phrasebank.csv")
    
    # If already downloaded, load from cache
    if os.path.exists(phrasebank_path):
        try:
            return pd.read_csv(phrasebank_path)
        except Exception:
            pass
    
    # Try to download Financial PhraseBank
    # Note: The actual dataset URL may vary. We'll try a common source.
    urls = [
        "https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis",
        # Alternative: Use a sample or create synthetic data if download fails
    ]
    
    # For now, we'll create a synthetic dataset based on common financial phrases
    # In production, you would download the actual dataset
    print("Note: Creating synthetic financial sentiment dataset.")
    print("For production use, download Financial PhraseBank from:")
    print("https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis")
    
    synthetic_data = _create_synthetic_financial_dataset()
    
    # Save for future use
    synthetic_data.to_csv(phrasebank_path, index=False)
    
    return synthetic_data


def _create_synthetic_financial_dataset() -> pd.DataFrame:
    """
    Create a synthetic financial sentiment dataset for demonstration.
    In production, replace with actual Financial PhraseBank download.
    """
    # Sample financial phrases with labels
    # 0: very negative, 1: negative, 2: neutral, 3: positive, 4: very positive
    samples = [
        # Very Negative (0)
        ("The company reported massive losses and bankruptcy is imminent", 0),
        ("Stock price crashed 50% following fraud allegations", 0),
        ("Severe decline in revenue threatens company survival", 0),
        ("Major layoffs announced as company struggles", 0),
        ("Regulatory investigation reveals serious violations", 0),
        
        # Negative (1)
        ("Earnings missed expectations causing stock decline", 1),
        ("Revenue decreased compared to last quarter", 1),
        ("Company faces challenges in competitive market", 1),
        ("Profit margins declined due to rising costs", 1),
        ("Market uncertainty affects company performance", 1),
        
        # Neutral (2)
        ("Company maintains steady performance this quarter", 2),
        ("No significant changes reported in financials", 2),
        ("Stock price remains stable around current levels", 2),
        ("Company continues normal operations", 2),
        ("Quarterly results align with market expectations", 2),
        
        # Positive (3)
        ("Strong earnings beat analyst expectations", 3),
        ("Revenue growth driven by new product launches", 3),
        ("Company expands market share in key segments", 3),
        ("Positive outlook for next quarter", 3),
        ("Strategic partnerships boost company prospects", 3),
        
        # Very Positive (4)
        ("Record-breaking profits exceed all forecasts", 4),
        ("Stock surges 30% on exceptional earnings report", 4),
        ("Company achieves market leadership position", 4),
        ("Breakthrough innovation drives massive growth", 4),
        ("Outstanding performance sets new industry standards", 4),
    ]
    
    # Expand with variations
    expanded_samples = []
    for text, label in samples:
        expanded_samples.append((text, label))
        # Add some variations
        variations = [
            text.replace("company", "firm"),
            text.replace("company", "corporation"),
            text.replace("stock", "share"),
            text.replace("revenue", "sales"),
        ]
        for var_text in variations[:2]:  # Limit variations
            expanded_samples.append((var_text, label))
    
    df = pd.DataFrame(expanded_samples, columns=["text", "label"])
    return df


def label_news_with_vader_scores(headlines: List[Dict[str, str]]) -> List[Dict[str, any]]:
    """
    Label news headlines using VADER sentiment scores as proxy labels.
    
    Maps VADER compound scores to 5 classes:
    - [-1, -0.6): Very Negative (0)
    - [-0.6, -0.2): Negative (1)
    - [-0.2, 0.2]: Neutral (2)
    - (0.2, 0.6]: Positive (3)
    - (0.6, 1]: Very Positive (4)
    
    Parameters
    ----------
    headlines : List[Dict[str, str]]
        List of headlines with 'text' key.
    
    Returns
    -------
    List[Dict[str, any]]
        List with 'text' and 'label' keys.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        print("Warning: vaderSentiment not available. Using simple heuristic.")
        analyzer = None
    
    labeled = []
    for item in headlines:
        text = item.get("text", "")
        if not text:
            continue
        
        if analyzer:
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
        else:
            # Simple heuristic fallback
            compound = _simple_sentiment_heuristic(text)
        
        # Map to 5 classes
        if compound < -0.6:
            label = 0  # Very Negative
        elif compound < -0.2:
            label = 1  # Negative
        elif compound <= 0.2:
            label = 2  # Neutral
        elif compound <= 0.6:
            label = 3  # Positive
        else:
            label = 4  # Very Positive
        
        labeled.append({
            "text": text,
            "label": label,
            "ticker": item.get("ticker", ""),
        })
    
    return labeled


def _simple_sentiment_heuristic(text: str) -> float:
    """Simple heuristic for sentiment when VADER is unavailable."""
    text_lower = text.lower()
    
    very_negative_words = ["crash", "bankruptcy", "fraud", "collapse", "disaster"]
    negative_words = ["decline", "loss", "miss", "fall", "drop", "down"]
    positive_words = ["growth", "gain", "rise", "beat", "surge", "up"]
    very_positive_words = ["record", "breakthrough", "exceptional", "outstanding"]
    
    score = 0.0
    for word in very_negative_words:
        if word in text_lower:
            score -= 0.8
    for word in negative_words:
        if word in text_lower:
            score -= 0.4
    for word in positive_words:
        if word in text_lower:
            score += 0.4
    for word in very_positive_words:
        if word in text_lower:
            score += 0.8
    
    return max(-1.0, min(1.0, score))


def combine_datasets(
    phrasebank_df: pd.DataFrame,
    news_headlines: List[Dict[str, any]],
    phrasebank_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Combine Financial PhraseBank and collected news headlines.
    
    Parameters
    ----------
    phrasebank_df : pd.DataFrame
        Financial PhraseBank dataset.
    news_headlines : List[Dict[str, any]]
        Labeled news headlines.
    phrasebank_weight : float
        Proportion of data to use from phrasebank (default 0.7).
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with 'text' and 'label' columns.
    """
    # Convert phrasebank to list
    phrasebank_list = []
    for _, row in phrasebank_df.iterrows():
        phrasebank_list.append({
            "text": str(row["text"]),
            "label": int(row["label"]),
        })
    
    # Sample phrasebank data if needed
    if phrasebank_weight < 1.0:
        n_phrasebank = int(len(phrasebank_list) * phrasebank_weight)
        import random
        random.shuffle(phrasebank_list)
        phrasebank_list = phrasebank_list[:n_phrasebank]
    
    # Combine
    combined = phrasebank_list + news_headlines
    
    # Shuffle
    import random
    random.shuffle(combined)
    
    df = pd.DataFrame(combined)
    return df


def create_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset.
    train_ratio : float
        Proportion for training set.
    val_ratio : float
        Proportion for validation set.
    test_ratio : float
        Proportion for test set.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df["label"] if "label" in df.columns else None,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=temp_df["label"] if "label" in temp_df.columns else None,
    )
    
    return train_df, val_df, test_df


def prepare_sentiment_datasets(
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to prepare all sentiment datasets.
    
    Parameters
    ----------
    use_cache : bool
        Whether to use cached datasets if available.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """
    data_dir = _data_dir()
    cache_path = os.path.join(data_dir, "sentiment_splits_cache.json")
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            train_df = pd.DataFrame(cache_data["train"])
            val_df = pd.DataFrame(cache_data["val"])
            test_df = pd.DataFrame(cache_data["test"])
            print(f"Loaded cached datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            return train_df, val_df, test_df
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    # Collect data
    print("Collecting news headlines from cache...")
    news_headlines = collect_news_headlines_from_cache()
    print(f"Collected {len(news_headlines)} news headlines")
    
    print("Labeling news headlines...")
    labeled_news = label_news_with_vader_scores(news_headlines)
    print(f"Labeled {len(labeled_news)} headlines")
    
    print("Downloading/preparing Financial PhraseBank...")
    phrasebank_df = download_financial_phrasebank()
    print(f"Financial PhraseBank: {len(phrasebank_df)} samples")
    
    print("Combining datasets...")
    combined_df = combine_datasets(phrasebank_df, labeled_news)
    print(f"Combined dataset: {len(combined_df)} samples")
    
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_splits(combined_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save to cache
    try:
        cache_data = {
            "train": train_df.to_dict("records"),
            "val": val_df.to_dict("records"),
            "test": test_df.to_dict("records"),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        print("Saved datasets to cache")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test data preparation
    train_df, val_df, test_df = prepare_sentiment_datasets()
    print("\nLabel distribution:")
    print("Train:", train_df["label"].value_counts().sort_index())
    print("Val:", val_df["label"].value_counts().sort_index())
    print("Test:", test_df["label"].value_counts().sort_index())

