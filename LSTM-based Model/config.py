from dataclasses import dataclass

@dataclass
class LSTMASLConfig:
    
    n_keypoints: int = 399
    vocab_size: int = 1

    batch_size: int = 16
    max_epochs: int = 50
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    conv_channels: int = 256
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.3