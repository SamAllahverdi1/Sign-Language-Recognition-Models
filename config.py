from dataclasses import dataclass, field

# special tokens

# for ctc, index 0
BLANK_TOKEN = '<blank>'
# for words < min_freq, should be at idnex 1
UNK_TOKEN   = '<unk>'


# just gonna use all landmarks since it barely increases latency and increases acc
# face mesh = 468*3 = 1404
# pose = 33 * 3 = 99
# left hand/right hand = 21*3 = 63 each

N_LANDMARKS = 543
N_KEYPOINTS = N_LANDMARKS * 3  # 1629


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
