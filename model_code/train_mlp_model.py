"""
Training script for the Lab 2–style MLP model.

It builds a dataset from historical price data (and optional sentiment/
fundamentals), trains the MLP to predict future N-day returns, evaluates
on a validation set, and saves weights to saved_models/stock_mlp.pth.
"""

import os
import sys
from datetime import datetime, timedelta
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
from domain.configs import MLPConfig
from domain.stocks import Stock, StockPriceSeries
from model_code.lab2_mlp_model import StockMLP, _build_tabular_features


class ReturnDataset(Dataset):
    """
    Each sample:
    - x: tabular features built from a rolling 30-day window ending at day t
    - y: class label for future N-day return from day t to t+N:
        0 = 下跌 (return <= -threshold)
        1 = 橫向/小波動 (|return| < threshold)
        2 = 上升 (return >= +threshold)
    """

    def __init__(
        self,
        series_list: List[StockPriceSeries],
        horizon_days: int = 5,
    ):
        self.features: List[np.ndarray] = []
        self.targets: List[int] = []
        threshold = 0.01  # 1% move considered significant

        for series in series_list:
            prices = series.prices
            if len(prices) < 40:
                continue

            # For simplicity we use a single sentiment / fundamentals snapshot per stock
            sentiment = fetch_sentiment_score(series.stock)
            fundamentals = fetch_fundamental_snapshot(series.stock)

            closes = np.array([p.close for p in prices], dtype=np.float32)

            # We require at least 30 days behind and horizon_days ahead
            for t in range(30, len(prices) - horizon_days):
                window_series = StockPriceSeries(
                    stock=series.stock, prices=prices[t - 30 : t]
                )
                feats, _ = _build_tabular_features(
                    window_series, sentiment=sentiment, fundamentals=fundamentals
                )

                future_return = (closes[t + horizon_days] / closes[t]) - 1.0

                # Discretise into 3 classes
                if future_return <= -threshold:
                    label = 0  # down
                elif future_return >= threshold:
                    label = 2  # up
                else:
                    label = 1  # flat

                self.features.append(feats.astype(np.float32))
                self.targets.append(int(label))

        self.features = np.stack(self.features, axis=0)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])
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


def train_mlp(
    epochs: int = 50,
    batch_size: int = 64,
    horizon_days: int = 5,
    lr: float = 1e-3,
    patience: int = 5,
) -> None:
    print("Preparing dataset...")
    series_list = build_series_for_watchlist(lookback_days=365)
    dataset = ReturnDataset(series_list, horizon_days=horizon_days)
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

    # Small hyperparameter sweep over MLPConfig
    search_space = [
        MLPConfig(input_dim=8, hidden_dim=32, num_layers=2),
        MLPConfig(input_dim=8, hidden_dim=64, num_layers=2),
        MLPConfig(input_dim=8, hidden_dim=128, num_layers=2),
        MLPConfig(input_dim=8, hidden_dim=64, num_layers=3),
    ]

    overall_best_loss = float("inf")
    overall_best_state = None
    overall_best_config: MLPConfig | None = None

    for cfg in search_space:
        print(f"=== Training MLP with config: hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers} ===")
        model = StockMLP(cfg, num_classes=3)
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
                f"[hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers}] "
                f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.6f}  val loss: {val_loss:.6f}"
            )

            # Early stopping similar to Lab 2: stop if val loss doesn't improve
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
            f"Finished config hidden_dim={cfg.hidden_dim}, layers={cfg.num_layers} "
            f"with best val loss={best_val_loss:.6f}"
        )
        if best_val_loss < overall_best_loss:
            overall_best_loss = best_val_loss
            overall_best_state = best_state
            overall_best_config = cfg

    if overall_best_state is None:
        print("No successful training run; not saving any weights.")
        return

    os.makedirs("saved_models", exist_ok=True)
    out_path = os.path.join("saved_models", "stock_mlp.pth")
    # Save both the config and the weights so that the loader can rebuild
    # the correct architecture later.
    checkpoint = {
        "config": {
            "input_dim": overall_best_config.input_dim,  # type: ignore[union-attr]
            "hidden_dim": overall_best_config.hidden_dim,  # type: ignore[union-attr]
            "num_layers": overall_best_config.num_layers,  # type: ignore[union-attr]
        },
        "state_dict": overall_best_state,
    }
    torch.save(checkpoint, out_path)
    print(
        f"Saved best MLP weights to {out_path} "
        f"(val loss={overall_best_loss:.6f}, "
        f"hidden_dim={overall_best_config.hidden_dim}, num_layers={overall_best_config.num_layers})"
    )


if __name__ == "__main__":
    train_mlp()


