import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from domain.configs import MLPConfig
from domain.stocks import PredictionInput, PredictionOutput, StockPriceSeries


class StockMLP(nn.Module):
    """
    Lab 2–style MLP for tabular stock features.

    Input: a 1D feature vector (technical indicators, sentiment, fundamentals).
    Output: logits for 3 classes:
        0 = 下跌 (down), 1 = 橫向/小波動 (flat), 2 = 上升 (up)
    """

    def __init__(self, config: MLPConfig, num_classes: int = 3):
        super().__init__()
        layers = []
        dim_in = config.input_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim_in, config.hidden_dim))
            layers.append(nn.ReLU())
            dim_in = config.hidden_dim
        layers.append(nn.Linear(dim_in, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) -> (batch, num_classes) logits
        return self.net(x)


def _build_tabular_features(
    series: StockPriceSeries,
    sentiment: Optional[float],
    fundamentals: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    """
    Build a small tabular feature vector from price history + sentiment + fundamentals.

    Features (example, length = 8):
    [last_close, ma_10, ma_30, std_10, std_30, sentiment_or_0, pe_or_0, ps_or_0]
    """

    closes = np.array([p.close for p in series.prices], dtype=np.float32)
    if len(closes) < 30:
        # pad by repeating last value to get stable statistics
        closes = np.pad(closes, (30 - len(closes), 0), mode="edge")

    last_close = closes[-1]
    ma_10 = closes[-10:].mean()
    ma_30 = closes[-30:].mean()
    std_10 = closes[-10:].std()
    std_30 = closes[-30:].std()

    sent = float(sentiment) if sentiment is not None else 0.0
    pe = float(fundamentals.get("pe_ratio", 0.0))
    ps = float(fundamentals.get("ps_ratio", 0.0))

    feats = np.array(
        [last_close, ma_10, ma_30, std_10, std_30, sent, pe, ps],
        dtype=np.float32,
    )
    return feats, last_close


def load_mlp_model(
    config: MLPConfig, weights_path: str
) -> Optional[StockMLP]:
    """
    Load a trained StockMLP if the weights file exists.
    """

    if not os.path.exists(weights_path):
        return None

    model = StockMLP(config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_with_mlp(
    series: StockPriceSeries,
    prediction_input: PredictionInput,
    sentiment: Optional[float],
    fundamentals: Dict[str, float],
    weights_path: str = "models/stock_mlp.pth",
) -> Optional[PredictionOutput]:
    """
    Run the Lab 2–style MLP, if trained weights are available.

    Returns a PredictionOutput or None if the model file is missing.
    """

    config = MLPConfig()
    model = load_mlp_model(config, weights_path)
    if model is None:
        return None

    feats, last_close = _build_tabular_features(series, sentiment, fundamentals)
    x = torch.from_numpy(feats).unsqueeze(0)  # (1, input_dim)

    with torch.no_grad():
        logits = model(x)  # (1, 3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Class mapping: 0 = down, 1 = flat, 2 = up
    class_idx = int(np.argmax(probs))
    prob = float(probs[class_idx])

    if class_idx == 2:
        expected_direction = "up"
        should_buy = True
        should_sell = False
        # simple heuristic: aim for +5% target, buy slightly below last close
        suggested_buy = last_close * 0.98
        suggested_sell = last_close * 1.05
    elif class_idx == 0:
        expected_direction = "down"
        should_buy = False
        should_sell = True
        # defensive: suggest taking profit slightly below last close
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


