"""
LSTM/GRU-based sentiment classification model.

Supports both random and pre-trained embeddings.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from ml.sentiment_preprocessing import Vocabulary


class SentimentLSTM(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.5,
        use_gru: bool = False,
        embedding_matrix: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        embedding_dim : int
            Embedding dimension.
        hidden_dim : int
            Hidden dimension of LSTM/GRU.
        num_layers : int
            Number of LSTM/GRU layers.
        num_classes : int
            Number of output classes (default: 5 for fine-grained sentiment).
        dropout : float
            Dropout rate.
        use_gru : bool
            Whether to use GRU instead of LSTM.
        embedding_matrix : Optional[np.ndarray]
            Pre-trained embedding matrix (vocab_size, embedding_dim).
            If None, embeddings are randomly initialized.
        freeze_embeddings : bool
            Whether to freeze embedding weights (only used with pre-trained embeddings).
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Embedding layer
        if embedding_matrix is not None:
            # Use pre-trained embeddings
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix,
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            # Random embeddings
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0,
            )
        
        # RNN layer (LSTM or GRU)
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape (batch_size, seq_len).
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN
        rnn_out, _ = self.rnn(embedded)  # (batch_size, seq_len, hidden_dim)
        
        # Use the last time step
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Classification
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)  # (batch_size, num_classes)
        
        return logits


class SentimentGRU(SentimentLSTM):
    """GRU-based sentiment classifier (convenience class)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize GRU model."""
        kwargs["use_gru"] = True
        super().__init__(*args, **kwargs)


def create_model(
    vocab: Vocabulary,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.5,
    use_gru: bool = False,
    embedding_matrix: Optional[np.ndarray] = None,
    freeze_embeddings: bool = False,
) -> SentimentLSTM:
    """
    Create sentiment model.
    
    Parameters
    ----------
    vocab : Vocabulary
        Vocabulary.
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
    embedding_matrix : Optional[np.ndarray]
        Pre-trained embedding matrix.
    freeze_embeddings : bool
        Whether to freeze embeddings.
    
    Returns
    -------
    SentimentLSTM
        Model instance.
    """
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        use_gru=use_gru,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

