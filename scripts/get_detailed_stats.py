#!/usr/bin/env python3
"""
Get detailed statistics for all models including per-class metrics.
"""

import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.price_data import fetch_fundamental_snapshot, fetch_price_history, fetch_sentiment_score
from domain.configs import MLPConfig, TransformerConfig
from domain.stocks import Stock, StockPriceSeries
from model_code.baseline_model import run_baseline_model
from model_code.lab2_mlp_model import StockMLP, _build_tabular_features
from model_code.lab5_transformer_model import StockTransformer, _build_sequence_features
from model_code.train_mlp_model import ReturnDataset
from model_code.train_transformer_model import SequenceReturnDataset
from datetime import datetime


def build_series_for_watchlist(lookback_days: int = 365):
    """Build price series for all stocks in watchlist"""
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
    series_list = []
    for s in watchlist:
        print(f"Fetching history for {s.ticker}...")
        try:
            series = fetch_price_history(s, end_date=end_date, lookback_days=lookback_days)
            series_list.append(series)
        except Exception as e:
            print(f"  Warning: Failed to fetch {s.ticker}: {e}")
    return series_list


def evaluate_mlp_detailed(dataset, model_path="saved_models/stock_mlp.pth"):
    """Evaluate MLP model with detailed metrics"""
    print("\n" + "="*60)
    print("MLP MODEL DETAILED EVALUATION")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = MLPConfig(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers']
    )
    model = StockMLP(config, num_classes=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Create test loader (use last 20% of data)
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    test_indices = list(range(total_size - test_size, total_size))
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\nPer-Class Performance:")
    class_names = ['Down (Class 0)', 'Flat (Class 1)', 'Up (Class 2)']
    for i, name in enumerate(class_names):
        if str(i) in report:
            print(f"  {name}: Precision {report[str(i)]['precision']:.2f}, "
                  f"Recall {report[str(i)]['recall']:.2f}, "
                  f"F1 {report[str(i)]['f1-score']:.2f}, "
                  f"Support: {report[str(i)]['support']}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Down  Flat   Up")
    for i, name in enumerate(['Down', 'Flat', 'Up']):
        print(f"Actual {name:4s}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}")
    
    print(f"\nBest Configuration: hidden_dim={checkpoint['config']['hidden_dim']}, "
          f"num_layers={checkpoint['config']['num_layers']}, "
          f"validation loss={checkpoint.get('best_val_loss', 'N/A')}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'config': checkpoint['config']
    }


def evaluate_transformer_detailed(dataset, model_path="saved_models/stock_transformer.pth"):
    """Evaluate Transformer model with detailed metrics"""
    print("\n" + "="*60)
    print("TRANSFORMER MODEL DETAILED EVALUATION")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = TransformerConfig(
        d_model=checkpoint['config']['d_model'],
        nhead=checkpoint['config']['nhead'],
        num_layers=checkpoint['config']['num_layers'],
        dim_feedforward=checkpoint['config']['dim_feedforward']
    )
    model = StockTransformer(feature_dim=8, config=config, num_classes=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Create test loader
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    test_indices = list(range(total_size - test_size, total_size))
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Recall: {report['macro avg']['recall']:.4f}")
    print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    print(f"\nPer-Class Performance:")
    class_names = ['Down (Class 0)', 'Flat (Class 1)', 'Up (Class 2)']
    for i, name in enumerate(class_names):
        if str(i) in report:
            print(f"  {name}: Precision {report[str(i)]['precision']:.2f}, "
                  f"Recall {report[str(i)]['recall']:.2f}, "
                  f"F1 {report[str(i)]['f1-score']:.2f}, "
                  f"Support: {report[str(i)]['support']}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Down  Flat   Up")
    for i, name in enumerate(['Down', 'Flat', 'Up']):
        print(f"Actual {name:4s}  {cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}")
    
    print(f"\nBest Configuration: d_model={checkpoint['config']['d_model']}, "
          f"nhead={checkpoint['config']['nhead']}, "
          f"num_layers={checkpoint['config']['num_layers']}, "
          f"dim_feedforward={checkpoint['config']['dim_feedforward']}, "
          f"validation loss={checkpoint.get('best_val_loss', 'N/A')}")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'config': checkpoint['config']
    }


def main():
    print("="*60)
    print("DETAILED MODEL STATISTICS")
    print("="*60)
    
    # Build datasets
    print("\nBuilding datasets...")
    series_list = build_series_for_watchlist(lookback_days=365)
    
    mlp_dataset = ReturnDataset(series_list, horizon_days=5)
    transformer_dataset = SequenceReturnDataset(series_list, window=30, horizon_days=5)
    
    print(f"MLP dataset size: {len(mlp_dataset)}")
    print(f"Transformer dataset size: {len(transformer_dataset)}")
    
    # Evaluate MLP
    if os.path.exists("saved_models/stock_mlp.pth"):
        mlp_results = evaluate_mlp_detailed(mlp_dataset)
    else:
        print("\nMLP model not found!")
    
    # Evaluate Transformer
    if os.path.exists("saved_models/stock_transformer.pth"):
        transformer_results = evaluate_transformer_detailed(transformer_dataset)
    else:
        print("\nTransformer model not found!")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if os.path.exists("saved_models/stock_mlp.pth"):
        print(f"\nMLP Model:")
        print(f"  Test Accuracy: {mlp_results['accuracy']:.4f} ({mlp_results['accuracy']*100:.2f}%)")
        print(f"  Macro Precision: {mlp_results['report']['macro avg']['precision']:.4f}")
        print(f"  Macro Recall: {mlp_results['report']['macro avg']['recall']:.4f}")
        print(f"  Macro F1: {mlp_results['report']['macro avg']['f1-score']:.4f}")
    
    if os.path.exists("saved_models/stock_transformer.pth"):
        print(f"\nTransformer Model:")
        print(f"  Test Accuracy: {transformer_results['accuracy']:.4f} ({transformer_results['accuracy']*100:.2f}%)")
        print(f"  Macro Precision: {transformer_results['report']['macro avg']['precision']:.4f}")
        print(f"  Macro Recall: {transformer_results['report']['macro avg']['recall']:.4f}")
        print(f"  Macro F1: {transformer_results['report']['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()

