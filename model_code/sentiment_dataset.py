"""
PyTorch Dataset classes for sentiment analysis.
"""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from model_code.sentiment_preprocessing import TextPreprocessor


class SentimentDataset(Dataset):
    """Dataset for sentiment classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: TextPreprocessor,
        fit_preprocessor: bool = False,
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings.
        labels : List[int]
            List of labels (0-4 for 5-class classification).
        preprocessor : TextPreprocessor
            Text preprocessor.
        fit_preprocessor : bool
            Whether to fit preprocessor on this data (use True for training set only).
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
        if fit_preprocessor:
            self.preprocessor.fit(texts)
        
        # Preprocess all texts
        self.sequences = self.preprocessor.transform(texts)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (sequence, label) tuple.
        """
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_len: int = 128,
    min_freq: int = 1,
) -> Tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """
    Create train/val/test datasets.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with 'text' and 'label' columns.
    val_df : pd.DataFrame
        Validation DataFrame.
    test_df : pd.DataFrame
        Test DataFrame.
    max_len : int
        Maximum sequence length.
    min_freq : int
        Minimum word frequency.
    
    Returns
    -------
    Tuple[SentimentDataset, SentimentDataset, SentimentDataset]
        Train, validation, and test datasets.
    """
    # Create preprocessor
    preprocessor = TextPreprocessor(max_len=max_len, min_freq=min_freq)
    
    # Create datasets
    train_dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        preprocessor=preprocessor,
        fit_preprocessor=True,  # Fit on training data only
    )
    
    val_dataset = SentimentDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        preprocessor=preprocessor,
        fit_preprocessor=False,
    )
    
    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        preprocessor=preprocessor,
        fit_preprocessor=False,
    )
    
    return train_dataset, val_dataset, test_dataset

