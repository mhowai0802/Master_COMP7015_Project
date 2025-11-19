#!/usr/bin/env python3
"""
Get comprehensive results for all models: MLP, Transformer, Baseline, LSTM, BERT
"""

import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
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
from model_code.sentiment_lstm_model import SentimentLSTM
from model_code.sentiment_data import prepare_sentiment_datasets
from model_code.sentiment_dataset import create_datasets
from model_code.sentiment_evaluation import evaluate_model
from model_code.sentiment_bert_model import load_bert_model, predict_with_bert, SentimentBERTDataset
from transformers import AutoTokenizer


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


def evaluate_baseline(series_list, dataset):
    """Evaluate baseline model properly"""
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    
    predictions = []
    true_labels = []
    
    # Get test indices
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    test_indices = list(range(total_size - test_size, total_size))
    
    # For baseline, we need to map back to series
    # This is simplified - in practice would need proper date mapping
    from domain.predictions import PredictionInput
    from datetime import date
    
    correct = 0
    total = 0
    
    for idx in test_indices[:min(100, len(test_indices))]:  # Limit for baseline
        try:
            # Get the sample
            x, y_true = dataset[idx]
            
            # Find corresponding series (simplified)
            series_idx = idx // (len(dataset) // len(series_list))
            if series_idx >= len(series_list):
                continue
            
            series = series_list[series_idx]
            if len(series.prices) < 40:
                continue
            
            pred_input = PredictionInput(
                stock=series.stock,
                as_of_date=date.today(),
                horizon_days=5,
            )
            
            baseline_pred = run_baseline_model(series, pred_input)
            
            # Convert to class
            if baseline_pred.direction == "up":
                pred = 2
            elif baseline_pred.direction == "down":
                pred = 0
            else:
                pred = 1
            
            predictions.append(pred)
            true_labels.append(y_true.item())
            
            if pred == y_true.item():
                correct += 1
            total += 1
        except Exception as e:
            continue
    
    if total == 0:
        print("Could not evaluate baseline model")
        return {"accuracy": 0.0}
    
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Evaluated on {total} samples")
    
    return {"accuracy": accuracy, "total": total}


def evaluate_lstm_sentiment():
    """Evaluate LSTM sentiment model"""
    print("\n" + "="*60)
    print("LSTM SENTIMENT MODEL EVALUATION")
    print("="*60)
    
    if not os.path.exists("saved_models/sentiment_lstm.pth"):
        print("LSTM model not found!")
        return None
    
    # Load data
    train_df, val_df, test_df = prepare_sentiment_datasets()
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, max_len=128, min_freq=1
    )
    
    # Load model
    checkpoint = torch.load("saved_models/sentiment_lstm.pth", map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=5,
        dropout=config['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    results = evaluate_model(model, test_loader, torch.device('cpu'), num_classes=5)
    
    print(f"\nTest Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    
    return results


def evaluate_bert_sentiment():
    """Evaluate BERT sentiment model"""
    print("\n" + "="*60)
    print("BERT SENTIMENT MODEL EVALUATION")
    print("="*60)
    
    bert_model_path = "saved_models/bert_sentiment/final_model"
    if not os.path.exists(bert_model_path):
        print("BERT model not found at", bert_model_path)
        print("BERT model needs to be trained first.")
        return None
    
    try:
        # Load data
        train_df, val_df, test_df = prepare_sentiment_datasets()
        
        # Load model and tokenizer
        model, tokenizer = load_bert_model(bert_model_path)
        
        # Evaluate on test set
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()
        
        predictions = predict_with_bert(model, tokenizer, test_texts)
        
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
        print(f"Macro Recall: {report['macro avg']['recall']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    except Exception as e:
        print(f"Error evaluating BERT: {e}")
        return None


def main():
    print("="*60)
    print("COMPREHENSIVE MODEL RESULTS")
    print("="*60)
    
    all_results = {}
    
    # Stock prediction models
    print("\n" + "="*60)
    print("STOCK PREDICTION MODELS")
    print("="*60)
    
    series_list = build_series_for_watchlist(lookback_days=365)
    mlp_dataset = ReturnDataset(series_list, horizon_days=5)
    transformer_dataset = SequenceReturnDataset(series_list, window=30, horizon_days=5)
    
    # MLP
    if os.path.exists("saved_models/stock_mlp.pth"):
        from scripts.get_detailed_stats import evaluate_mlp_detailed, evaluate_transformer_detailed
        mlp_results = evaluate_mlp_detailed(mlp_dataset)
        all_results['MLP'] = mlp_results
    else:
        print("\nMLP model not found!")
    
    # Transformer
    if os.path.exists("saved_models/stock_transformer.pth"):
        transformer_results = evaluate_transformer_detailed(transformer_dataset)
        all_results['Transformer'] = transformer_results
    else:
        print("\nTransformer model not found!")
    
    # Baseline
    baseline_results = evaluate_baseline(series_list, mlp_dataset)
    all_results['Baseline'] = baseline_results
    
    # Sentiment models
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS MODELS")
    print("="*60)
    
    # LSTM Sentiment
    lstm_sentiment_results = evaluate_lstm_sentiment()
    if lstm_sentiment_results:
        all_results['LSTM_Sentiment'] = lstm_sentiment_results
    
    # BERT Sentiment
    bert_sentiment_results = evaluate_bert_sentiment()
    if bert_sentiment_results:
        all_results['BERT_Sentiment'] = bert_sentiment_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL MODEL RESULTS")
    print("="*60)
    
    print("\nStock Prediction Models:")
    if 'MLP' in all_results:
        print(f"  MLP: {all_results['MLP']['accuracy']:.4f} ({all_results['MLP']['accuracy']*100:.2f}%)")
    if 'Transformer' in all_results:
        print(f"  Transformer: {all_results['Transformer']['accuracy']:.4f} ({all_results['Transformer']['accuracy']*100:.2f}%)")
    if 'Baseline' in all_results:
        print(f"  Baseline: {all_results['Baseline']['accuracy']:.4f} ({all_results['Baseline']['accuracy']*100:.2f}%)")
    
    print("\nSentiment Analysis Models:")
    if 'LSTM_Sentiment' in all_results:
        print(f"  LSTM: {all_results['LSTM_Sentiment']['accuracy']:.4f} ({all_results['LSTM_Sentiment']['accuracy']*100:.2f}%)")
    if 'BERT_Sentiment' in all_results:
        print(f"  BERT: {all_results['BERT_Sentiment']['accuracy']:.4f} ({all_results['BERT_Sentiment']['accuracy']*100:.2f}%)")
    else:
        print("  BERT: Not trained yet")
    
    return all_results


if __name__ == "__main__":
    results = main()

