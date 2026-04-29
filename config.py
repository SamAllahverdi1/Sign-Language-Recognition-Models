from dataclasses import dataclass


@dataclass
class TransformerASLConfig:
    n_keypoints: int = 399
    vocab_size: int = 1

    batch_size: int = 16
    max_epochs: int = 100
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    warmup_epochs: int = 10

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
