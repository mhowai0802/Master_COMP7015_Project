#!/usr/bin/env python3
"""
Training script for BERT sentiment analysis model.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_code.sentiment_data import prepare_sentiment_datasets
from model_code.sentiment_bert_model import SentimentBERTDataset, train_bert_model
from transformers import AutoTokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BERT sentiment analysis model")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                       help="BERT model name (default: bert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=str, default="saved_models/bert_sentiment")
    
    args = parser.parse_args()
    
    # Prepare data
    print("Preparing datasets...")
    train_df, val_df, test_df = prepare_sentiment_datasets()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = SentimentBERTDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    val_dataset = SentimentBERTDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    test_dataset = SentimentBERTDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Train
    results = train_bert_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model_name=args.model_name,
        num_classes=5,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        output_dir=args.output,
        patience=args.patience,
    )
    
    print("\nTraining completed!")
    if results.get('val_results'):
        print(f"Validation accuracy: {results['val_results'].get('eval_accuracy', 'N/A')}")
    if results.get('test_results'):
        print(f"Test accuracy: {results['test_results'].get('eval_accuracy', 'N/A')}")


if __name__ == "__main__":
    main()

