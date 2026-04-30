import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from config import TransformerASLConfig
from Transformer_ASL import TransformerASL

# google drive paths
TRAIN_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.train"
DEV_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.dev"
TEST_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.test"
CKPT_DIR = "/content/drive/MyDrive/ASL Translation Project - CS 4641/checkpoints_phoenix"

PHOENIX_KEYPOINT_FEATURES = 133 * 3
MAX_PHOENIX_FRAMES = 300

def build_phoenix_vocab(train_keypoint_data: dict) -> tuple[dict, dict]:
    gloss_counts = Counter()

    for phoenix_clip in train_keypoint_data.values():
        gloss_counts.update(phoenix_clip["gloss"].split())

    phoenix_vocab = ["<blank>", "<unk>"]
    phoenix_vocab += sorted(gloss_counts.keys())

    gloss_to_index = {gloss: index for index, gloss in enumerate(phoenix_vocab)}
    index_to_gloss = {index: gloss for index, gloss in enumerate(phoenix_vocab)}

    return gloss_to_index, index_to_gloss


def ctc_greedy_tokens(gloss_log_probs: torch.Tensor, blank_index: int = 0) -> list[int]:
    best_frame_indices = gloss_log_probs.argmax(dim=-1).tolist()
    decoded_tokens = []
    previous_index = None

    for gloss_index in best_frame_indices:
        repeated_gloss = gloss_index == previous_index
        blank_gloss = gloss_index == blank_index

        # skip if same token or blank
        if not repeated_gloss and not blank_gloss:
            decoded_tokens.append(gloss_index)

        previous_index = gloss_index

    return decoded_tokens

# translate all into german
def tokens_to_gloss(gloss_tokens: list[int], index_to_gloss: dict) -> str:
    decoded_glosses = [
        index_to_gloss.get(gloss_token, "<unk>")
        for gloss_token in gloss_tokens
    ]

    return " ".join(decoded_glosses)

class PhoenixDataset(Dataset):
    def __init__(self, phoenix_data_path: str, gloss_to_index: dict):
        raw_phoenix_file = np.load(phoenix_data_path, allow_pickle=True)
        all_phoenix_clips = list(dict(raw_phoenix_file).values())

        self.phoenix_clips = all_phoenix_clips
        self.gloss_to_index = gloss_to_index

        print(f"Loaded {len(self.phoenix_clips)} PHOENIX clips ")

    def __len__(self):
        return len(self.phoenix_clips)

    def __getitem__(self, clip_index):
        phoenix_clip = self.phoenix_clips[clip_index]
        raw_keypoints = phoenix_clip["keypoint"]

        if isinstance(raw_keypoints, torch.Tensor):
            keypoint_frames = raw_keypoints.float()
        else:
            keypoint_frames = torch.from_numpy(raw_keypoints.astype(np.float32))

        keypoint_frames = keypoint_frames.reshape(keypoint_frames.shape[0], -1)

        # normalize
        keypoint_mean = keypoint_frames.mean()
        keypoint_std = keypoint_frames.std()
        keypoint_frames = (keypoint_frames - keypoint_mean) / (keypoint_std + 1e-6)

        keypoint_frames = keypoint_frames[:MAX_PHOENIX_FRAMES]

        gloss_tokens = [self.gloss_to_index.get(gloss, 1) for gloss in phoenix_clip["gloss"].split()]
        gloss_labels = torch.tensor(gloss_tokens, dtype=torch.long)

        return keypoint_frames, gloss_labels


def collate_phoenix_batch(batch):
    keypoint_sequences, gloss_sequences = zip(*batch)

    keypoint_lengths = torch.tensor([sequence.shape[0] for sequence in keypoint_sequences], dtype=torch.long)
    padded_keypoints = pad_sequence(keypoint_sequences, batch_first=True)

    gloss_lengths = torch.tensor([sequence.shape[0] for sequence in gloss_sequences], dtype=torch.long)
    combined_gloss_labels = torch.cat(gloss_sequences)

    return (
        padded_keypoints,
        keypoint_lengths,
        combined_gloss_labels,
        gloss_lengths,
    )


def make_padding_mask(padded_keypoints: torch.Tensor, keypoint_lengths: torch.Tensor) -> torch.Tensor:
    
    longest_clip = padded_keypoints.size(1)
    frame_numbers = torch.arange(longest_clip, device=padded_keypoints.device)

    return frame_numbers.unsqueeze(0) >= keypoint_lengths.unsqueeze(1)


def run_one_epoch(
    model: TransformerASL,
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
            gloss_labels,
            gloss_lengths,
        ) = batch

        padded_keypoints = padded_keypoints.to(device)
        keypoint_lengths = keypoint_lengths.to(device)
        gloss_labels = gloss_labels.to(device)

        padding_mask = make_padding_mask(
            padded_keypoints,
            keypoint_lengths,
        )

        optimizer.zero_grad()

        gloss_log_probs = model(
            padded_keypoints,
            src_key_padding_mask=padding_mask,
        )

        loss = ctc_loss(
            gloss_log_probs,
            gloss_labels,
            keypoint_lengths.cpu(),
            gloss_lengths,
        )

        if not torch.isfinite(loss):
            print("Skipped one PHOENIX batch because the CTC loss broke.")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_train_loss += loss.item()
        trained_batches += 1

    return total_train_loss / max(trained_batches, 1)


def run_validation(
    model: TransformerASL,
    phoenix_loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    index_to_gloss: dict,
    device: torch.device,
    epoch: int,
) -> tuple[float, list[tuple[str, str]]]:
    
    model.eval()
    total_val_loss = 0.0
    sample_predictions = []

    # no_grad = no param update
    with torch.no_grad():
        for batch in tqdm(phoenix_loader, desc=f"Epoch {epoch} val"):
            (
                padded_keypoints,
                keypoint_lengths,
                gloss_labels,
                gloss_lengths,
            ) = batch

            padded_keypoints = padded_keypoints.to(device)
            keypoint_lengths = keypoint_lengths.to(device)
            gloss_labels = gloss_labels.to(device)

            padding_mask = make_padding_mask(padded_keypoints, keypoint_lengths)

            gloss_log_probs = model(padded_keypoints, src_key_padding_mask=padding_mask)
            loss = ctc_loss(gloss_log_probs, gloss_labels, keypoint_lengths.cpu(), gloss_lengths)

            total_val_loss += loss.item()

            label_start = 0

            for batch_index in range(padded_keypoints.size(0)):
                if len(sample_predictions) >= 3:
                    break

                clip_frame_count = keypoint_lengths[batch_index].item()
                predicted_tokens = ctc_greedy_tokens(gloss_log_probs[:clip_frame_count, batch_index, :])

                label_count = gloss_lengths[batch_index].item()
                label_end = label_start + label_count
                target_tokens = gloss_labels[label_start:label_end].tolist()

                sample_predictions.append((tokens_to_gloss(target_tokens, index_to_gloss), tokens_to_gloss(predicted_tokens, index_to_gloss)))

                label_start = label_end

    average_val_loss = total_val_loss / max(len(phoenix_loader), 1)

    return average_val_loss, sample_predictions


def train(
    max_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 10,
    weight_decay: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading PHOENIX-2014T keypoints...")
    train_keypoint_data = dict(
        np.load(TRAIN_KP, allow_pickle=True) # must allow pickle since files are pkl
    )
    gloss_to_index, index_to_gloss = build_phoenix_vocab(train_keypoint_data)

    print(f"PHOENIX gloss vocabulary: {len(gloss_to_index)} tokens")

    cfg = TransformerASLConfig(
        n_keypoints=PHOENIX_KEYPOINT_FEATURES,
        vocab_size=len(gloss_to_index),
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
    )

    train_dataset = PhoenixDataset(TRAIN_KP, gloss_to_index)
    dev_dataset = PhoenixDataset(DEV_KP, gloss_to_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_phoenix_batch,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_phoenix_batch,
        num_workers=4,
        pin_memory=True,
    )

    model = TransformerASL(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    ctc_loss = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input keypoints: {PHOENIX_KEYPOINT_FEATURES}")
    print(f"Learning rate stays fixed at {learning_rate}")

    for epoch in range(1, max_epochs + 1):
        train_loss = run_one_epoch(
            model,
            train_loader,
            optimizer,
            ctc_loss,
            grad_clip,
            device,
            epoch,
        )
        val_loss, sample_predictions = run_validation(
            model,
            dev_loader,
            ctc_loss,
            index_to_gloss,
            device,
            epoch,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f}"
        )

        for sample_number, prediction in enumerate(sample_predictions, start=1):
            reference_gloss, predicted_gloss = prediction
            print(f"  Sample {sample_number} REF: {reference_gloss}")
            print(f"  Sample {sample_number} HYP: {predicted_gloss}")

        checkpoint = {
            "epoch": epoch,
            "cfg": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gloss_to_idx": gloss_to_index,
            "idx_to_gloss": index_to_gloss,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(CKPT_DIR, "phoenix_best.pt")
            torch.save(checkpoint, checkpoint_path)
            print("  Saved the best PHOENIX checkpoint so far.")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")

    return best_val_loss


if __name__ == "__main__":
    train()