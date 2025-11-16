"""
Training script for the Lab 5–style Transformer model.

It builds sequence datasets from historical price data (plus sentiment/
fundamentals), trains the Transformer to predict future N-day returns,
evaluates on a validation set, and saves weights to models/stock_transformer.pth.
"""

import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Ensure project root is on sys.path so that `api`, `ml`, and `domain` are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.price_data import (
    fetch_fundamental_snapshot,
    fetch_price_history,
    fetch_sentiment_score,
)
from domain.configs import TransformerConfig
from domain.stocks import Stock, StockPriceSeries
from ml.lab5_transformer_model import (
    StockTransformer,
    _build_sequence_features,
)


class SequenceReturnDataset(Dataset):
    """
    Each sample:
    - x: sequence of daily features for a fixed window (e.g. 30 days)
    - y: class label for future N-day return from last day in window to last+N:
        0 = 下跌 (return <= -threshold)
        1 = 橫向/小波動 (|return| < threshold)
        2 = 上升 (return >= +threshold)
    """

    def __init__(
        self,
        series_list: List[StockPriceSeries],
        window: int = 30,
        horizon_days: int = 5,
    ):
        self.sequences: List[np.ndarray] = []
        self.targets: List[int] = []
        threshold = 0.01  # 1% move

        for series in series_list:
            prices = series.prices
            if len(prices) < window + horizon_days + 1:
                continue

            sentiment = fetch_sentiment_score(series.stock)
            fundamentals = fetch_fundamental_snapshot(series.stock)
            closes = np.array([p.close for p in prices], dtype=np.float32)

            for t in range(window, len(prices) - horizon_days):
                window_series = StockPriceSeries(
                    stock=series.stock, prices=prices[t - window : t]
                )
                feats = _build_sequence_features(
                    window_series,
                    sentiment=sentiment,
                    fundamentals=fundamentals,
                    max_len=window,
                )
                future_return = (closes[t + horizon_days] / closes[t]) - 1.0
                if future_return <= -threshold:
                    label = 0
                elif future_return >= threshold:
                    label = 2
                else:
                    label = 1
                self.sequences.append(feats.astype(np.float32))
                self.targets.append(int(label))

        self.sequences = np.stack(self.sequences, axis=0)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.sequences[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


def build_series_for_watchlist(lookback_days: int = 365) -> List[StockPriceSeries]:
    watchlist = [
        Stock(name="Apple", ticker="AAPL"),
        Stock(name="Microsoft", ticker="MSFT"),
        Stock(name="NVIDIA", ticker="NVDA"),
        Stock(name="Alphabet (Google)", ticker="GOOGL"),
        Stock(name="Amazon", ticker="AMZN"),
        Stock(name="Meta", ticker="META"),
        Stock(name="Tesla", ticker="TSLA"),
        Stock(name="Broadcom", ticker="AVGO"),
        Stock(name="TSMC", ticker="TSM"),
        Stock(name="Super Micro Computer", ticker="SMCI"),
    ]

    end_date = datetime.today()
    series_list: List[StockPriceSeries] = []
    for s in watchlist:
        print(f"Fetching history for {s.ticker} ...")
        series = fetch_price_history(s, end_date=end_date, lookback_days=lookback_days)
        series_list.append(series)
    return series_list


def train_transformer(
    epochs: int = 50,
    batch_size: int = 64,
    window: int = 30,
    horizon_days: int = 5,
    lr: float = 1e-3,
    patience: int = 5,
) -> None:
    print("Preparing sequence dataset...")
    series_list = build_series_for_watchlist(lookback_days=365)
    dataset = SequenceReturnDataset(
        series_list, window=window, horizon_days=horizon_days
    )
    print(f"Total samples: {len(dataset)}")

    if len(dataset) < 100:
        print("Not enough samples to train a meaningful model.")
        return

    # Train/val split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Small hyperparameter sweep over TransformerConfig
    search_space = [
        TransformerConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64),
        TransformerConfig(d_model=64, nhead=4, num_layers=2, dim_feedforward=128),
        TransformerConfig(d_model=64, nhead=8, num_layers=3, dim_feedforward=128),
    ]

    feature_dim = 8  # open, high, low, close, volume, sentiment, pe, ps

    overall_best_loss = float("inf")
    overall_best_state = None
    overall_best_config: TransformerConfig | None = None

    for cfg in search_space:
        print(
            "=== Training Transformer with config: "
            f"d_model={cfg.d_model}, nhead={cfg.nhead}, layers={cfg.num_layers}, "
            f"dim_feedforward={cfg.dim_feedforward} ==="
        )

        model = StockTransformer(feature_dim=feature_dim, config=cfg, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                # x: (batch, seq_len, feature_dim)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / train_size

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= val_size

            print(
                f"[d_model={cfg.d_model}, nhead={cfg.nhead}, layers={cfg.num_layers}] "
                f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.6f}  val loss: {val_loss:.6f}"
            )

            # Early stopping similar to Lab 5: stop if val loss doesn't improve
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = model.state_dict()
                no_improve = 0
                print("  -> New best model for this config (val loss improved)")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("  -> Early stopping for this config.")
                    break

        if best_state is None:
            continue

        print(
            "Finished config "
            f"d_model={cfg.d_model}, nhead={cfg.nhead}, layers={cfg.num_layers}, "
            f"dim_feedforward={cfg.dim_feedforward} with best val loss={best_val_loss:.6f}"
        )

        if best_val_loss < overall_best_loss:
            overall_best_loss = best_val_loss
            overall_best_state = best_state
            overall_best_config = cfg

    if overall_best_state is None:
        print("No successful training run; not saving any weights.")
        return

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "stock_transformer.pth")
    checkpoint = {
        "config": {
            "d_model": overall_best_config.d_model,  # type: ignore[union-attr]
            "nhead": overall_best_config.nhead,  # type: ignore[union-attr]
            "num_layers": overall_best_config.num_layers,  # type: ignore[union-attr]
            "dim_feedforward": overall_best_config.dim_feedforward,  # type: ignore[union-attr]
            "dropout": overall_best_config.dropout,  # type: ignore[union-attr]
            "max_len": overall_best_config.max_len,  # type: ignore[union-attr]
        },
        "state_dict": overall_best_state,
    }
    torch.save(checkpoint, out_path)
    print(
        f"Saved best Transformer weights to {out_path} "
        f"(val loss={overall_best_loss:.6f}, "
        f"d_model={overall_best_config.d_model}, nhead={overall_best_config.nhead}, "
        f"num_layers={overall_best_config.num_layers}, "
        f"dim_feedforward={overall_best_config.dim_feedforward})"
    )


if __name__ == "__main__":
    train_transformer()


