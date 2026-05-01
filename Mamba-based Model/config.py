from dataclasses import dataclass, field

# special tokens

# for ctc, index 0
BLANK_TOKEN = '<blank>'
# for words < min_freq, should be at idnex 1
UNK_TOKEN   = '<unk>'


# body joints = 17*3 = 51       joints  0-16  (17 joints * 3 = 51)
# feet = 6*3 = 18
# face = 68*3 = 204
# left/right hand = 21 * 3 = 63 heac joints 91-111 (21 joints * 3 = 63)

N_LANDMARKS = 133
N_KEYPOINTS = N_LANDMARKS * 3

@dataclass
class MambaASLConfig:
    n_keypoints: int = N_KEYPOINTS
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4         
    dt_rank: int = 16      
    expand: int = 2  
    # might not need this it should get overwritten
    vocab_size: int = 2
    # training dims
    dropout: float = 0.3
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_epochs: int = 100
    grad_clip: float = 1.0
    # data
    keypoints_dir: str = "data/keypoints"
    splits_dir: str = "data/splits"
