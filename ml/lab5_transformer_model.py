import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from domain.configs import TransformerConfig
from domain.stocks import PredictionInput, PredictionOutput, StockPriceSeries


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
    Output: a regression scalar (expected future return over horizon).
    """

    def __init__(self, feature_dim: int, config: TransformerConfig):
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
            nn.Linear(config.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        # Use the last time step as summary (like using CLS token)
        last = h[:, -1, :]  # (batch, d_model)
        out = self.head(last).squeeze(-1)  # (batch,)
        return out


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
    config: TransformerConfig,
    weights_path: str,
) -> Optional[StockTransformer]:
    """
    Load a trained StockTransformer if the weights file exists.
    """

    if not os.path.exists(weights_path):
        return None

    model = StockTransformer(feature_dim=feature_dim, config=config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_with_transformer(
    series: StockPriceSeries,
    prediction_input: PredictionInput,
    sentiment: Optional[float],
    fundamentals: Dict[str, float],
    weights_path: str = "models/stock_transformer.pth",
) -> Optional[PredictionOutput]:
    """
    Run the Lab 5–style Transformer, if trained weights are available.

    Returns a PredictionOutput or None if the model file is missing.
    """

    config = TransformerConfig()
    feature_dim = 8  # open, high, low, close, volume, sentiment, pe, ps
    model = load_transformer_model(feature_dim, config, weights_path)
    if model is None:
        return None

    feats = _build_sequence_features(
        series, sentiment=sentiment, fundamentals=fundamentals, max_len=config.max_len
    )
    x = torch.from_numpy(feats).unsqueeze(0)  # (1, seq_len, feature_dim)

    with torch.no_grad():
        future_return = float(model(x).item())

    last_close = float(series.prices[-1].close)
    expected_direction = "up" if future_return > 0.01 else "down" if future_return < -0.01 else "flat"
    confidence = min(0.99, float(abs(future_return)))

    suggested_buy = last_close * (1.0 + min(future_return, 0) - 0.02)
    suggested_sell = last_close * (1.0 + max(future_return, 0) + 0.03)

    should_buy = expected_direction == "up"
    should_sell = expected_direction == "down"

    return PredictionOutput(
        should_buy=should_buy,
        should_sell=should_sell,
        expected_direction=expected_direction,
        suggested_buy_price=suggested_buy,
        suggested_sell_price=suggested_sell,
        confidence=confidence,
    )


# Training suggestion:
# - Construct sequences of daily features for many stocks over many windows.
# - Use the future N-day return as regression target.
# - Train with MSE or Huber loss as in Lab 5 attention examples.
# - Save weights to 'models/stock_transformer.pth' and the frontend will load it.


