"""
Minimal script to create a dummy (untrained) MLP weight file
so that the frontend can load and display Lab 2 MLP results.
"""

import os

import torch

from domain.configs import MLPConfig
from ml.lab2_mlp_model import StockMLP


def main() -> None:
    os.makedirs("models", exist_ok=True)

    config = MLPConfig()
    model = StockMLP(config)

    out_path = os.path.join("models", "stock_mlp.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved dummy MLP weights to {out_path}")


if __name__ == "__main__":
    main()


