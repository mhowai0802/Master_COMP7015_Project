from dataclasses import dataclass


@dataclass
class MLPConfig:
    """
    Configuration for the Lab 2–style MLP model.
    """

    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 2


@dataclass
class TransformerConfig:
    """
    Configuration for the Lab 5–style Transformer model.
    """

    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1
    max_len: int = 128


