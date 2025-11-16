"""
Minimal script to create a dummy (untrained) Transformer weight file
so that the frontend can load and display Lab 5 Transformer results.
"""

import os

import torch

from domain.configs import TransformerConfig
from ml.lab5_transformer_model import StockTransformer


def main() -> None:
    os.makedirs("models", exist_ok=True)

    config = TransformerConfig()
    feature_dim = 8  # must match lab5_transformer_model.py
    model = StockTransformer(feature_dim=feature_dim, config=config)

    out_path = os.path.join("models", "stock_transformer.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved dummy Transformer weights to {out_path}")


if __name__ == "__main__":
    main()


