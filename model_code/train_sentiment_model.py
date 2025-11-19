"""
Training script for sentiment analysis models.

Supports hyperparameter experimentation and comparison of random vs pre-trained embeddings.
"""

import os
import json
from typing import Dict, Optional, Tuple
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from model_code.sentiment_data import prepare_sentiment_datasets
from model_code.sentiment_dataset import create_datasets
from model_code.sentiment_lstm_model import create_model, count_parameters
from model_code.sentiment_preprocessing import (
    TextPreprocessor,
    load_glove_embeddings,
    create_embedding_matrix,
    Vocabulary,
)
from model_code.sentiment_evaluation import evaluate_model, print_evaluation_report


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns
    -------
    Tuple[float, float]
        (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate model.
    
    Returns
    -------
    Tuple[float, float]
        (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def train_sentiment_model(
    train_dataset,
    val_dataset,
    test_dataset,
    vocab: Vocabulary,
    embedding_type: str = "random",  # "random" or "glove"
    glove_path: Optional[str] = None,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.5,
    use_gru: bool = False,
    freeze_embeddings: bool = False,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    patience: int = 5,
    device: Optional[torch.device] = None,
    model_save_path: Optional[str] = None,
) -> Dict:
    """
    Train sentiment model.
    
    Parameters
    ----------
    train_dataset : SentimentDataset
        Training dataset.
    val_dataset : SentimentDataset
        Validation dataset.
    test_dataset : SentimentDataset
        Test dataset.
    vocab : Vocabulary
        Vocabulary.
    embedding_type : str
        "random" or "glove".
    glove_path : Optional[str]
        Path to GloVe embeddings file.
    embedding_dim : int
        Embedding dimension.
    hidden_dim : int
        Hidden dimension.
    num_layers : int
        Number of RNN layers.
    num_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    use_gru : bool
        Whether to use GRU.
    freeze_embeddings : bool
        Whether to freeze embeddings.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    epochs : int
        Maximum number of epochs.
    patience : int
        Early stopping patience.
    device : Optional[torch.device]
        Device to use.
    model_save_path : Optional[str]
        Path to save best model.
    
    Returns
    -------
    Dict
        Training history and best model info.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Embedding type: {embedding_type}")
    
    # Prepare embedding matrix
    embedding_matrix = None
    if embedding_type == "glove" and glove_path:
        print(f"Loading GloVe embeddings from {glove_path}...")
        glove_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)
        if glove_embeddings:
            embedding_matrix = create_embedding_matrix(vocab, glove_embeddings, embedding_dim)
            embedding_matrix = np.array(embedding_matrix)
            print(f"Loaded {len(glove_embeddings)} word embeddings")
        else:
            print("Warning: Could not load GloVe embeddings, using random embeddings")
            embedding_type = "random"
    
    # Create model
    model = create_model(
        vocab=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        use_gru=use_gru,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
    )
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "embedding_type": embedding_type,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_classes": num_classes,
                    "dropout": dropout,
                    "use_gru": use_gru,
                    "config": {
                        "max_len": train_dataset.preprocessor.max_len,
                        "embedding_dim": embedding_dim,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "num_classes": num_classes,
                        "dropout": dropout,
                        "use_gru": use_gru,
                    },
                }, model_save_path)
                print(f"Saved best model to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for testing
    if model_save_path and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded best model for testing")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, device, num_classes)
    print_evaluation_report(test_results, "Test")
    
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_results": test_results,
        "model_config": {
            "embedding_type": embedding_type,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
            "use_gru": use_gru,
        },
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--embedding", type=str, default="random", choices=["random", "glove"])
    parser.add_argument("--glove-path", type=str, default=None)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-gru", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--output", type=str, default="saved_models/sentiment_lstm.pth")
    
    args = parser.parse_args()
    
    # Prepare data
    print("Preparing datasets...")
    train_df, val_df, test_df = prepare_sentiment_datasets()
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df,
        max_len=args.max_len,
        min_freq=args.min_freq,
    )
    
    vocab = train_dataset.preprocessor.vocab
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train
    results = train_sentiment_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
        embedding_type=args.embedding,
        glove_path=args.glove_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gru=args.use_gru,
        freeze_embeddings=args.freeze_embeddings,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        model_save_path=args.output,
    )
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()

