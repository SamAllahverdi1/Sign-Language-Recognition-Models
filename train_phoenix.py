import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TransformerASLConfig
from Transformer_ASL import TransformerASL

TRAIN_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.train"
DEV_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.dev"
TEST_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.test"
CKPT_DIR = "/content/drive/MyDrive/ASL Translation Project - CS 4641/checkpoints_phoenix"

N_KEYPOINTS = 133 * 3
MAX_FRAMES = 300


def build_phoenix_vocab(train_data: dict) -> tuple[dict, dict]:
    counts = Counter()
    for item in train_data.values():
        counts.update(item["gloss"].split())
    vocab = ["<blank>", "<unk>"] + sorted(counts.keys())
    gloss_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_gloss = {idx: token for idx, token in enumerate(vocab)}
    return gloss_to_idx, idx_to_gloss


def ctc_greedy_tokens(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    indices = log_probs.argmax(-1).tolist()
    tokens = []
    prev = None
    for idx in indices:
        if idx != prev and idx != blank:
            tokens.append(idx)
        prev = idx
    return tokens


def tokens_to_gloss(tokens: list[int], idx_to_gloss: dict) -> str:
    return " ".join(idx_to_gloss.get(token, "<unk>") for token in tokens)


class PhoenixDataset(Dataset):
    def __init__(self, data_path: str, gloss_to_idx: dict):
        raw = np.load(data_path, allow_pickle=True)
        items = list(dict(raw).values())

        valid = []
        for item in items:
            n_frames = min(item["keypoint"].shape[0], MAX_FRAMES)
            n_glosses = len(item["gloss"].split())
            if n_frames >= max(1, 2 * n_glosses - 1):
                valid.append(item)

        dropped = len(items) - len(valid)
        self.items = valid
        self.gloss_to_idx = gloss_to_idx
        print(f"Loaded {len(self.items)} clips ({dropped} dropped - CTC too short)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        keypoints = item["keypoint"]

        if isinstance(keypoints, torch.Tensor):
            frames = keypoints.float().reshape(keypoints.shape[0], -1)
        else:
            frames = torch.from_numpy(keypoints.astype(np.float32)).reshape(keypoints.shape[0], -1)

        # normalize
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        
        frames = frames[:MAX_FRAMES]
        labels = torch.tensor(
            [self.gloss_to_idx.get(token, 1) for token in item["gloss"].split()],
            dtype=torch.long,
        )
        return frames, labels


def collate_fn(batch):
    frames, labels = zip(*batch)
    frame_lengths = torch.tensor([frame.shape[0] for frame in frames], dtype=torch.long)
    frames_padded = pad_sequence(frames, batch_first=True)
    label_lengths = torch.tensor([label.shape[0] for label in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return frames_padded, frame_lengths, labels_concat, label_lengths


def build_scheduler(optimizer, max_epochs: int, warmup_epochs: int):
    warmup_epochs = min(max(warmup_epochs, 0), max_epochs)
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    cosine_epochs = max(max_epochs - warmup_epochs, 1)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def train(
    max_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    patience: int = 10,
    weight_decay: float = 0.0,
    warmup_epochs: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Loading PHOENIX keypoints...")
    train_raw = dict(np.load(TRAIN_KP, allow_pickle=True))
    gloss_to_idx, idx_to_gloss = build_phoenix_vocab(train_raw)
    print(f"Gloss vocabulary: {len(gloss_to_idx)} tokens")

    cfg = TransformerASLConfig()
    cfg.n_keypoints = N_KEYPOINTS
    cfg.vocab_size = len(gloss_to_idx)
    cfg.batch_size = batch_size
    cfg.max_epochs = max_epochs
    cfg.learning_rate = learning_rate
    cfg.grad_clip = grad_clip
    cfg.weight_decay = weight_decay
    cfg.warmup_epochs = warmup_epochs

    train_ds = PhoenixDataset(TRAIN_KP, gloss_to_idx)
    dev_ds = PhoenixDataset(DEV_KP, gloss_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = TransformerASL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, max_epochs, warmup_epochs)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    best_val = float("inf")
    epochs_no_improve = 0

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input dim: {N_KEYPOINTS} | Vocab: {cfg.vocab_size} | LR: {learning_rate}")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for frames, frame_lens, labels, label_lens in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            frames = frames.to(device)
            labels = labels.to(device)
            frame_lens = frame_lens.to(device)

            max_len = frames.size(1)
            padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= frame_lens.unsqueeze(1)

            optimizer.zero_grad()
            log_probs = model(frames, src_key_padding_mask=padding_mask)
            loss = ctc_loss(log_probs, labels, frame_lens.cpu(), label_lens)

            if not torch.isfinite(loss):
                print(f"Propagating NaN/Inf loss encountered, skipping batch...")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        sample_predictions = []

        with torch.no_grad():
            for frames, frame_lens, labels, label_lens in tqdm(dev_loader, desc=f"Epoch {epoch} val"):
                frames = frames.to(device)
                labels = labels.to(device)
                frame_lens = frame_lens.to(device)

                max_len = frames.size(1)
                padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= frame_lens.unsqueeze(1)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    log_probs = model(frames, src_key_padding_mask=padding_mask)
                    loss = ctc_loss(log_probs, labels, frame_lens.cpu(), label_lens)

                val_loss += loss.item()

                label_offset = 0
                for batch_idx in range(frames.size(0)):
                    if len(sample_predictions) >= 3:
                        break
                    seq_len = frame_lens[batch_idx].item()
                    pred_tokens = ctc_greedy_tokens(log_probs[:seq_len, batch_idx, :])
                    target_len = label_lens[batch_idx].item()
                    target_tokens = labels[label_offset:label_offset + target_len].tolist()
                    sample_predictions.append((
                        tokens_to_gloss(target_tokens, idx_to_gloss),
                        tokens_to_gloss(pred_tokens, idx_to_gloss),
                    ))
                    label_offset += target_len

        val_loss /= len(dev_loader)
        scheduler.step()
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6g}")
        for i, (reference, hypothesis) in enumerate(sample_predictions, start=1):
            print(f"  Sample {i} REF: {reference}")
            print(f"  Sample {i} HYP: {hypothesis}")

        checkpoint = {
            "epoch": epoch,
            "cfg": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gloss_to_idx": gloss_to_idx,
            "idx_to_gloss": idx_to_gloss,
        }

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(CKPT_DIR, "phoenix_best.pt"))
            print("  -> best checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Done. Best val loss: {best_val:.4f}")
    return best_val


if __name__ == "__main__":
    train()
