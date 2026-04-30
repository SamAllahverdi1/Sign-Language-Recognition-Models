import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from config import LSTMASLConfig
from LSTM_ASL import ConvBiLSTMCTC

TRAIN_KP = "/content/drive/MyDrive/ASL_Project/Phoenix-2014T.train"
DEV_KP = "/content/drive/MyDrive/ASL_Project/Phoenix-2014T.dev"
TEST_KP = "/content/drive/MyDrive/ASL_Project/Phoenix-2014T.test"
CKPT_DIR = "/content/drive/MyDrive/ASL_Project/checkpoints_phoenix"

PHOENIX_KEYPOINT_FEATURES = 133 * 3
MAX_PHOENIX_FRAMES = 300

def load_phoenix_pickle(phoenix_data_path: str) -> dict:
    with open(phoenix_data_path, "rb") as raw_phoenix_file:
        return pickle.load(raw_phoenix_file)

def build_phoenix_vocab(*phoenix_data_dicts: dict) -> tuple[dict, dict]:
    all_text_chars = ""

    for phoenix_data in phoenix_data_dicts:
        for phoenix_clip in phoenix_data.values():
            all_text_chars += phoenix_clip["text"].lower()

    sorted_chars = sorted(set(all_text_chars))

    phoenix_vocab = ["<blank>"] + sorted_chars
    char_to_index = {char: index for index, char in enumerate(phoenix_vocab)}
    index_to_char = {index: char for index, char in enumerate(phoenix_vocab)}

    return char_to_index, index_to_char

def compute_feature_stats(train_keypoint_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    all_frames = []

    for phoenix_clip in train_keypoint_data.values():
        keypoints = phoenix_clip["keypoint"].float()
        all_frames.append(keypoints.reshape(-1, PHOENIX_KEYPOINT_FEATURES))

    all_frames = torch.cat(all_frames, dim=0)
    feature_mean = all_frames.mean(dim=0)
    feature_std = all_frames.std(dim=0).clamp(min=1e-6)

    return feature_mean, feature_std


def ctc_greedy_tokens(token_log_probs: torch.Tensor, blank_index: int = 0) -> list[int]:
    best_frame_indices = token_log_probs.argmax(dim=-1).tolist()
    decoded_tokens = []
    previous_index = None

    for token_index in best_frame_indices:
        repeated_token = token_index == previous_index
        blank_token = token_index == blank_index

        if not repeated_token and not blank_token:
            decoded_tokens.append(token_index)

        previous_index = token_index

    return decoded_tokens

def tokens_to_text(token_indices: list[int], index_to_char: dict) -> str:
    return "".join(index_to_char.get(token_index, "") for token_index in token_indices)

class PhoenixDataset(Dataset):
    def __init__(
        self,
        phoenix_data_path: str,
        char_to_index: dict,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        raw_phoenix_data = load_phoenix_pickle(phoenix_data_path)
        self.phoenix_clips = list(raw_phoenix_data.values())
        self.char_to_index = char_to_index
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        print(f"Loaded {len(self.phoenix_clips)} PHOENIX clips")

    def __len__(self):
        return len(self.phoenix_clips)

    def __getitem__(self, clip_index):
        phoenix_clip = self.phoenix_clips[clip_index]

        keypoint_frames = phoenix_clip["keypoint"].float()

        if keypoint_frames.shape[0] > MAX_PHOENIX_FRAMES:
            keypoint_frames = keypoint_frames[:MAX_PHOENIX_FRAMES]

        keypoint_frames = keypoint_frames.reshape(keypoint_frames.shape[0], -1)
        keypoint_frames = (keypoint_frames - self.feature_mean) / self.feature_std

        clip_text = phoenix_clip["text"].lower().strip()
        text_token_indices = [
            self.char_to_index[character]
            for character in clip_text
            if character in self.char_to_index
        ]
        text_labels = torch.tensor(text_token_indices, dtype=torch.long)

        return keypoint_frames, text_labels

def collate_phoenix_batch(batch):
    keypoint_sequences, text_sequences = zip(*batch)

    keypoint_lengths = torch.tensor([sequence.shape[0] for sequence in keypoint_sequences], dtype=torch.long)
    padded_keypoints = pad_sequence(keypoint_sequences, batch_first=True)

    text_lengths = torch.tensor([sequence.shape[0] for sequence in text_sequences], dtype=torch.long)
    combined_text_labels = torch.cat(text_sequences)

    return (
        padded_keypoints,
        keypoint_lengths,
        combined_text_labels,
        text_lengths,
    )

def run_one_epoch(
    model: ConvBiLSTMCTC,
    phoenix_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss: nn.CTCLoss,
    grad_clip: float,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_train_loss = 0.0
    trained_batches = 0

    for batch in tqdm(phoenix_loader, desc=f"Epoch {epoch} train"):
        (
            padded_keypoints,
            keypoint_lengths,
            text_labels,
            text_lengths,
        ) = batch

        padded_keypoints = padded_keypoints.to(device)
        text_labels = text_labels.to(device)

        optimizer.zero_grad()

        text_log_probs = model(padded_keypoints)

        loss = ctc_loss(
            text_log_probs,
            text_labels,
            keypoint_lengths,
            text_lengths,
        )

        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_train_loss += loss.item()
        trained_batches += 1

    return total_train_loss / max(trained_batches, 1)


def run_validation(
    model: ConvBiLSTMCTC,
    phoenix_loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(phoenix_loader, desc=f"Epoch {epoch} val"):
            (
                padded_keypoints,
                keypoint_lengths,
                text_labels,
                text_lengths,
            ) = batch

            padded_keypoints = padded_keypoints.to(device)
            text_labels = text_labels.to(device)

            text_log_probs = model(padded_keypoints)
            loss = ctc_loss(text_log_probs, text_labels, keypoint_lengths, text_lengths)

            total_val_loss += loss.item()

    return total_val_loss / max(len(phoenix_loader), 1)

def train(
    max_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    grad_clip: float = 1.0,
    weight_decay: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CKPT_DIR, exist_ok=True)

    train_keypoint_data = load_phoenix_pickle(TRAIN_KP)
    dev_keypoint_data = load_phoenix_pickle(DEV_KP)

    char_to_index, index_to_char = build_phoenix_vocab(train_keypoint_data, dev_keypoint_data)
    feature_mean, feature_std = compute_feature_stats(train_keypoint_data)

    cfg = LSTMASLConfig(
        n_keypoints=PHOENIX_KEYPOINT_FEATURES,
        vocab_size=len(char_to_index),
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
    )

    train_dataset = PhoenixDataset(TRAIN_KP, char_to_index, feature_mean, feature_std)
    dev_dataset = PhoenixDataset(DEV_KP, char_to_index, feature_mean, feature_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_phoenix_batch,
        num_workers=2,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_phoenix_batch,
        num_workers=2,
        pin_memory=True,
    )

    model = ConvBiLSTMCTC(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_val_loss = float("inf")

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input keypoints: {PHOENIX_KEYPOINT_FEATURES}")

    for epoch in range(1, max_epochs + 1):
        
        train_loss = run_one_epoch(model, train_loader, optimizer, ctc_loss, grad_clip, device,epoch)
        val_loss = run_validation(model, dev_loader, ctc_loss, device, epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"lr={current_lr:.1e}"
        )

        checkpoint = {
            "epoch": epoch,
            "cfg": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "char_to_idx": char_to_index,
            "idx_to_char": index_to_char,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(CKPT_DIR, "phoenix_best.pt"))

    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_val_loss

if __name__ == "__main__":
    train()