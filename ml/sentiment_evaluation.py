"""
Evaluation metrics and reporting for sentiment analysis models.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Dict:
    """
    Evaluate model and compute metrics.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    dataloader : DataLoader
        Data loader.
    device : torch.device
        Device.
    num_classes : int
        Number of classes.
    
    Returns
    -------
    Dict
        Dictionary with evaluation metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            predictions = logits.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Weighted-averaged metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == i).sum() / mask.sum()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def print_evaluation_report(results: Dict, dataset_name: str = "Dataset") -> None:
    """
    Print evaluation report.
    
    Parameters
    ----------
    results : Dict
        Evaluation results dictionary.
    dataset_name : str
        Name of the dataset.
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {results['macro_precision']:.4f}")
    print(f"  Recall: {results['macro_recall']:.4f}")
    print(f"  F1-Score: {results['macro_f1']:.4f}")
    
    print(f"\nWeighted-averaged Metrics:")
    print(f"  Precision: {results['weighted_precision']:.4f}")
    print(f"  Recall: {results['weighted_recall']:.4f}")
    print(f"  F1-Score: {results['weighted_f1']:.4f}")
    
    print(f"\nPer-class Metrics:")
    class_names = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    for i, (name, prec, rec, f1, supp, acc) in enumerate(zip(
        class_names[:len(results['precision'])],
        results['precision'],
        results['recall'],
        results['f1'],
        results['support'],
        results['per_class_accuracy'],
    )):
        print(
            f"  {name:15s} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f} | "
            f"F1: {f1:.4f} | "
            f"Support: {supp:4d} | "
            f"Accuracy: {acc:.4f}"
        )
    
    print(f"\nConfusion Matrix:")
    print_confusion_matrix(results['confusion_matrix'], class_names[:len(results['precision'])])
    
    print(f"{'='*60}\n")


def print_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """
    Print confusion matrix in a readable format.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : List[str]
        List of class names.
    """
    n_classes = len(class_names)
    
    # Header
    print(" " * 15, end="")
    for name in class_names:
        print(f"{name[:8]:>10s}", end="")
    print()
    
    # Rows
    for i, name in enumerate(class_names):
        print(f"{name[:15]:15s}", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:10d}", end="")
        print()


def compare_models(results_random: Dict, results_pretrained: Dict) -> None:
    """
    Compare two models (e.g., random vs pre-trained embeddings).
    
    Parameters
    ----------
    results_random : Dict
        Results from model with random embeddings.
    results_pretrained : Dict
        Results from model with pre-trained embeddings.
    """
    print(f"\n{'='*60}")
    print("Model Comparison: Random Embeddings vs Pre-trained Embeddings")
    print(f"{'='*60}")
    
    print(f"\nOverall Accuracy:")
    print(f"  Random:      {results_random['accuracy']:.4f}")
    print(f"  Pre-trained: {results_pretrained['accuracy']:.4f}")
    print(f"  Improvement: {results_pretrained['accuracy'] - results_random['accuracy']:.4f}")
    
    print(f"\nMacro-averaged F1-Score:")
    print(f"  Random:      {results_random['macro_f1']:.4f}")
    print(f"  Pre-trained: {results_pretrained['macro_f1']:.4f}")
    print(f"  Improvement: {results_pretrained['macro_f1'] - results_random['macro_f1']:.4f}")
    
    print(f"\nWeighted-averaged F1-Score:")
    print(f"  Random:      {results_random['weighted_f1']:.4f}")
    print(f"  Pre-trained: {results_pretrained['weighted_f1']:.4f}")
    print(f"  Improvement: {results_pretrained['weighted_f1'] - results_random['weighted_f1']:.4f}")
    
    print(f"{'='*60}\n")

