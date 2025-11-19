import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from domain.configs import TransformerConfig
from domain.stocks import StockPriceSeries
from domain.predictions import PredictionInput, PredictionOutput


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class StockTransformer(nn.Module):
    """
    Lab 5–style Transformer encoder adapted for stock time series.

    Input: sequence of daily feature vectors (OHLCV + sentiment + fundamentals).
    Output: logits for 3 classes:
        0 = 下跌 (down), 1 = 橫向/小波動 (flat), 2 = 上升 (up)
    """

    def __init__(self, feature_dim: int, config: TransformerConfig, num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        # Use the last time step as summary (like using CLS token)
        last = h[:, -1, :]  # (batch, d_model)
        logits = self.head(last)  # (batch, num_classes)
        return logits


def _build_sequence_features(
    series: StockPriceSeries,
    sentiment: Optional[float],
    fundamentals: Dict[str, float],
    max_len: int,
) -> np.ndarray:
    """
    Build a 2D array (seq_len, feature_dim) with per-day features.

    Per-day features (example):
    [open, high, low, close, volume, sentiment, pe, ps]
    Sentiment and fundamentals are broadcast across all days.
    """

    sentiment_val = float(sentiment) if sentiment is not None else 0.0
    pe = float(fundamentals.get("pe_ratio", 0.0))
    ps = float(fundamentals.get("ps_ratio", 0.0))

    records = []
    for p in series.prices:
        records.append(
            [
                float(p.open),
                float(p.high),
                float(p.low),
                float(p.close),
                float(p.volume),
                sentiment_val,
                pe,
                ps,
            ]
        )

    feats = np.array(records, dtype=np.float32)
    # Keep only the most recent max_len days
    if feats.shape[0] > max_len:
        feats = feats[-max_len:, :]
    return feats


def load_transformer_model(
    feature_dim: int,
    weights_path: str,
) -> Optional[StockTransformer]:
    """
    Load a trained StockTransformer if the weights file exists.
    """

    if not os.path.exists(weights_path):
        return None

    state = torch.load(weights_path, map_location="cpu")

    # New-style checkpoints: {"config": {...}, "state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state and "config" in state:
        cfg_dict = state["config"]
        config = TransformerConfig(
            d_model=cfg_dict.get("d_model", 32),
            nhead=cfg_dict.get("nhead", 4),
            num_layers=cfg_dict.get("num_layers", 2),
            dim_feedforward=cfg_dict.get("dim_feedforward", 64),
            dropout=cfg_dict.get("dropout", 0.1),
            max_len=cfg_dict.get("max_len", 128),
        )
        model = StockTransformer(feature_dim=feature_dim, config=config, num_classes=3)
        model.load_state_dict(state["state_dict"])
    else:
        # Backward compatibility: old checkpoints were a bare state_dict
        config = TransformerConfig()
        model = StockTransformer(feature_dim=feature_dim, config=config, num_classes=3)
        try:
            model.load_state_dict(state)
        except Exception:
            # If shapes don't match (e.g. checkpoint from different architecture),
            # fail gracefully so the caller can treat it as "no trained model".
            return None

    model.eval()
    return model


def predict_with_transformer(
    series: StockPriceSeries,
    prediction_input: PredictionInput,
    sentiment: Optional[float],
    fundamentals: Dict[str, float],
    weights_path: str = "saved_models/stock_transformer.pth",
) -> Optional[PredictionOutput]:
    """
    Run the Lab 5–style Transformer, if trained weights are available.

    Returns a PredictionOutput or None if the model file is missing.
    """

    feature_dim = 8  # open, high, low, close, volume, sentiment, pe, ps
    model = load_transformer_model(feature_dim, weights_path)
    if model is None:
        return None

    # Use default max_len from config for feature window; the model itself
    # will be built with the correct max_len from the checkpoint when available.
    max_len = TransformerConfig().max_len
    feats = _build_sequence_features(
        series, sentiment=sentiment, fundamentals=fundamentals, max_len=max_len
    )
    x = torch.from_numpy(feats).unsqueeze(0)  # (1, seq_len, feature_dim)

    with torch.no_grad():
        logits = model(x)  # (1, 3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Class mapping: 0 = down, 1 = flat, 2 = up
    class_idx = int(np.argmax(probs))
    prob = float(probs[class_idx])

    last_close = float(series.prices[-1].close)
    if class_idx == 2:
        expected_direction = "up"
        should_buy = True
        should_sell = False
        suggested_buy = last_close * 0.98
        suggested_sell = last_close * 1.05
    elif class_idx == 0:
        expected_direction = "down"
        should_buy = False
        should_sell = True
        suggested_buy = last_close * 0.95
        suggested_sell = last_close * 0.98
    else:
        expected_direction = "flat"
        should_buy = False
        should_sell = False
        suggested_buy = last_close * 0.99
        suggested_sell = last_close * 1.01

    return PredictionOutput(
        should_buy=should_buy,
        should_sell=should_sell,
        expected_direction=expected_direction,
        suggested_buy_price=suggested_buy,
        suggested_sell_price=suggested_sell,
        confidence=prob,
    )


# Training suggestion:
# - Construct sequences of daily features for many stocks over many windows.
# - Use the future N-day return as regression target.
# - Train with MSE or Huber loss as in Lab 5 attention examples.
# - Save weights to 'saved_models/stock_transformer.pth' and the frontend will load it.


