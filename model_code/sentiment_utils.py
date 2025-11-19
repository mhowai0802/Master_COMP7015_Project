"""
Utility functions for loading sentiment models and making predictions.
"""

import os
from typing import Optional, List, Union
import torch
import torch.nn as nn
import numpy as np

from model_code.sentiment_preprocessing import TextPreprocessor, Vocabulary
from model_code.sentiment_lstm_model import SentimentLSTM, create_model
from model_code.sentiment_bert_model import load_bert_model, predict_with_bert


def load_lstm_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load trained LSTM model and preprocessor.
    
    Parameters
    ----------
    model_path : str
        Path to saved model checkpoint.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    tuple
        (model, preprocessor, config) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruct preprocessor
    vocab = checkpoint.get("vocab")
    if vocab is None:
        raise ValueError("Vocabulary not found in checkpoint")
    
    config = checkpoint.get("config", {})
    
    preprocessor = TextPreprocessor(
        vocab=vocab,
        max_len=config.get("max_len", 128),
    )
    
    # Reconstruct model
    model = create_model(
        vocab=vocab,
        embedding_dim=config.get("embedding_dim", 100),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        num_classes=config.get("num_classes", 5),
        dropout=config.get("dropout", 0.5),
        use_gru=config.get("use_gru", False),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, preprocessor, config


def predict_sentiment_lstm(
    model: nn.Module,
    preprocessor: TextPreprocessor,
    texts: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> Union[int, np.ndarray]:
    """
    Predict sentiment using LSTM model.
    
    Parameters
    ----------
    model : nn.Module
        Trained LSTM model.
    preprocessor : TextPreprocessor
        Text preprocessor.
    texts : Union[str, List[str]]
        Text or list of texts.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    Union[int, np.ndarray]
        Predicted class label(s).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    model.eval()
    
    # Preprocess
    sequences = preprocessor.transform(texts)
    sequences_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(sequences_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()
    
    if is_single:
        return int(predictions[0])
    return predictions


def predict_sentiment_bert(
    model_path: str,
    texts: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> Union[int, np.ndarray]:
    """
    Predict sentiment using BERT model.
    
    Parameters
    ----------
    model_path : str
        Path to saved BERT model.
    texts : Union[str, List[str]]
        Text or list of texts.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    Union[int, np.ndarray]
        Predicted class label(s).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    model, tokenizer = load_bert_model(model_path, device)
    predictions = predict_with_bert(model, tokenizer, texts, device)
    
    if is_single:
        return int(predictions[0])
    return predictions


def class_to_score(class_label: int, num_classes: int = 5) -> float:
    """
    Convert class label to continuous sentiment score in [-1, 1].
    
    Parameters
    ----------
    class_label : int
        Class label (0-4 for 5-class).
    num_classes : int
        Number of classes.
    
    Returns
    -------
    float
        Sentiment score in [-1, 1] range.
    """
    # Map 5 classes to [-1, 1]:
    # 0 (Very Negative) -> -1.0
    # 1 (Negative) -> -0.5
    # 2 (Neutral) -> 0.0
    # 3 (Positive) -> 0.5
    # 4 (Very Positive) -> 1.0
    
    if num_classes == 5:
        mapping = {
            0: -1.0,  # Very Negative
            1: -0.5,  # Negative
            2: 0.0,   # Neutral
            3: 0.5,   # Positive
            4: 1.0,   # Very Positive
        }
        return mapping.get(class_label, 0.0)
    elif num_classes == 2:
        # Binary classification
        return 1.0 if class_label == 1 else -1.0
    else:
        # Linear mapping for other numbers of classes
        return (class_label / (num_classes - 1)) * 2.0 - 1.0


def predict_sentiment_score(
    texts: Union[str, List[str]],
    model_type: str = "lstm",
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Union[float, List[float]]:
    """
    Predict sentiment score(s) for text(s).
    
    This is the main API function that returns continuous scores compatible
    with the existing stock prediction pipeline.
    
    Parameters
    ----------
    texts : Union[str, List[str]]
        Text or list of texts.
    model_type : str
        Model type: "lstm" or "bert" (default: "lstm").
    model_path : Optional[str]
        Path to model. If None, uses default paths.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    Union[float, List[float]]
        Sentiment score(s) in [-1, 1] range.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    # Default model paths
    if model_path is None:
        if model_type == "lstm":
            model_path = "saved_models/sentiment_lstm.pth"
        elif model_type == "bert":
            model_path = "saved_models/bert_sentiment/final_model"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Using fallback.")
        # Fallback: return neutral scores
        scores = [0.0] * len(texts)
        return scores[0] if is_single else scores
    
    # Predict
    if model_type == "lstm":
        model, preprocessor, config = load_lstm_model(model_path, device)
        predictions = predict_sentiment_lstm(model, preprocessor, texts, device)
        num_classes = config.get("num_classes", 5)
    elif model_type == "bert":
        predictions = predict_sentiment_bert(model_path, texts, device)
        num_classes = 5  # BERT model uses 5 classes
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Convert to scores
    if isinstance(predictions, np.ndarray):
        scores = [class_to_score(int(p), num_classes) for p in predictions]
    else:
        scores = [class_to_score(int(predictions), num_classes)]
    
    return scores[0] if is_single else scores

