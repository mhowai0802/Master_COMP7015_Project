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
    Output: a single scalar representing expected future return (regression),
            which we turn into direction / buy-sell logic.
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        layers = []
        dim_in = config.input_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim_in, config.hidden_dim))
            layers.append(nn.ReLU())
            dim_in = config.hidden_dim
        layers.append(nn.Linear(dim_in, 1))  # regression: future return
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        return self.net(x).squeeze(-1)  # (batch,)


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
        future_return = float(model(x).item())  # e.g. expected percentage move

    # Interpret regression output as relative change (e.g. 0.05 = +5%)
    expected_direction = "up" if future_return > 0.01 else "down" if future_return < -0.01 else "flat"
    confidence = min(0.99, float(abs(future_return)))  # simple proxy

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


# NOTE:
# To train this model, you would typically:
# - Build a dataset of (features, future_return) pairs from many stocks and days.
# - Use a regression loss (e.g. MSE) and the usual PyTorch training loop (see Lab 2).
# - Save weights to 'models/stock_mlp.pth' so the frontend can load and use it.


