"""
BERT fine-tuning for financial sentiment analysis.

Fine-tunes BERT or financial BERT variants for 5-class sentiment classification.
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
import numpy as np


class SentimentBERTDataset(Dataset):
    """Dataset for BERT fine-tuning."""
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Initialize dataset.
        
        Parameters
        ----------
        texts : list
            List of text strings.
        labels : list
            List of labels.
        tokenizer
            BERT tokenizer.
        max_length : int
            Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SentimentBERT(nn.Module):
    """BERT-based sentiment classifier."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        """
        Initialize BERT model.
        
        Parameters
        ----------
        model_name : str
            Hugging Face model name (default: "bert-base-uncased").
            Can use "yiyanghkust/finbert-pretrain" for financial BERT.
        num_classes : int
            Number of output classes.
        dropout : float
            Dropout rate.
        """
        super().__init__()
        
        # Use AutoModelForSequenceClassification for easier fine-tuning
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs.
        attention_mask : torch.Tensor, optional
            Attention mask.
        labels : torch.Tensor, optional
            Ground truth labels.
        
        Returns
        -------
        torch.Tensor or tuple
            Logits or (loss, logits) if labels provided.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def create_bert_model(
    model_name: str = "bert-base-uncased",
    num_classes: int = 5,
    dropout: float = 0.1,
) -> SentimentBERT:
    """
    Create BERT model.
    
    Parameters
    ----------
    model_name : str
        Model name.
    num_classes : int
        Number of classes.
    dropout : float
        Dropout rate.
    
    Returns
    -------
    SentimentBERT
        Model instance.
    """
    return SentimentBERT(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
    )


def train_bert_model(
    train_dataset: SentimentBERTDataset,
    val_dataset: SentimentBERTDataset,
    test_dataset: Optional[SentimentBERTDataset] = None,
    model_name: str = "bert-base-uncased",
    num_classes: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 10,
    max_length: int = 128,
    output_dir: str = "saved_models/bert_sentiment",
    patience: int = 3,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Fine-tune BERT model using Hugging Face Trainer.
    
    Parameters
    ----------
    train_dataset : SentimentBERTDataset
        Training dataset.
    val_dataset : SentimentBERTDataset
        Validation dataset.
    test_dataset : Optional[SentimentBERTDataset]
        Test dataset (optional).
    model_name : str
        BERT model name.
    num_classes : int
        Number of classes.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    epochs : int
        Number of epochs.
    max_length : int
        Maximum sequence length.
    output_dir : str
        Output directory for saving model.
    patience : int
        Early stopping patience.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    Dict
        Training results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create model
    model = create_bert_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )
    
    # Custom compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        
        return {
            "accuracy": accuracy,
            "f1": f1,
        }
    
    # Create trainer
    trainer = Trainer(
        model=model.model,  # Use the underlying model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    
    # Train
    print("Starting BERT fine-tuning...")
    train_result = trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = trainer.evaluate()
    
    # Evaluate on test set if provided
    test_results = None
    if test_dataset is not None:
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
    
    # Save final model
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    print(f"\nModel saved to {output_dir}/final_model")
    
    return {
        "train_results": train_result,
        "val_results": val_results,
        "test_results": test_results,
        "model_path": f"{output_dir}/final_model",
    }


def load_bert_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load trained BERT model and tokenizer.
    
    Parameters
    ----------
    model_path : str
        Path to saved model.
    device : Optional[torch.device]
        Device to use.
    
    Returns
    -------
    tuple
        (model, tokenizer) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def predict_with_bert(
    model,
    tokenizer,
    texts: list,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> np.ndarray:
    """
    Predict sentiment for a list of texts.
    
    Parameters
    ----------
    model
        Trained BERT model.
    tokenizer
        BERT tokenizer.
    texts : list
        List of text strings.
    device : Optional[torch.device]
        Device to use.
    max_length : int
        Maximum sequence length.
    
    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                str(text),
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = logits.argmax(dim=1).cpu().numpy()[0]
            predictions.append(pred)
    
    return np.array(predictions)

