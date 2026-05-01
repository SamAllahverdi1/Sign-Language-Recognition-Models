import os
import sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# paths for the Colab doc 
TRAIN_KP = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.train"
DEV_KP   = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.dev"
TEST_KP  = "/content/drive/MyDrive/ASL Translation Project - CS 4641/PhoenixDataset/Phoenix-2014T.test"
CKPT_DIR = "/content/drive/MyDrive/ASL Translation Project - CS 4641/checkpoints_phoenix"

#HRNet 133 landmarks * 3 for xyz 
N_KEYPOINTS = 133 * 3  # 399
MAX_FRAMES  = 300


def build_phoenix_vocab(train_data: dict) -> tuple[dict, dict]:
    # build gloss annotations
    counts = Counter()
    for item in train_data.values():
        counts.update(item['gloss'].split())
    glosses = sorted(counts.keys())
    vocab = ['<blank>', '<unk>'] + glosses
    gloss_to_idx = {g: i for i, g in enumerate(vocab)}
    idx_to_gloss = {i: g for i, g in enumerate(vocab)}
    return gloss_to_idx, idx_to_gloss


class PhoenixDataset(Dataset):
    # dropped shortest sequences for ctc constraint
    # limit is currently t >= 2l - 1
    def __init__(self, data_path: str, gloss_to_idx: dict):
        raw = np.load(data_path, allow_pickle=True)
        items = list(dict(raw).values())
        valid = []
        for item in items:
            n_frames = min(item['keypoint'].shape[0], MAX_FRAMES)
            n_glosses = len(item['gloss'].split())
            if n_frames>= max(1, 2 * n_glosses - 1):
                valid.append(item)
        n_dropped = len(items) -len(valid)
        self.items = valid
        self.gloss_to_idx = gloss_to_idx
        print(f"total clips: {len(self.items)} dropped clips: {n_dropped}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        kp   = item['keypoint']
        if isinstance(kp, torch.Tensor):
            # tensor should ne (T, 399)
            frames = kp.float().reshape(kp.shape[0], -1)
        else:
            frames = torch.from_numpy(kp.astype(np.float32)).reshape(kp.shape[0], -1)
        if frames.shape[0] > MAX_FRAMES:
            frames = frames[:MAX_FRAMES]
        label = torch.tensor(
            [self.gloss_to_idx.get(g, 1) for g in item['gloss'].split()],
            dtype=torch.long
        )
        return frames, label


def collate_fn(batch):
    frames, labels = zip(*batch)
    frame_lengths = torch.tensor([f.shape[0] for f in frames], dtype=torch.long)
    frames_padded = pad_sequence(frames, batch_first=True)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)
    return frames_padded, frame_lengths, labels_cat, label_lengths


def train(max_epochs: int = 100, batch_size: int = 16,
          learning_rate: float = 3e-5, grad_clip: float = 0.5,
          patience: int = 10, weight_decay: float = 0.01,
          n_layers: int = 6):

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.config import MambaASLConfig
    from model.mamba_asl_fast import MambaASLFast

    device = torch.device('cuda')
    os.makedirs(CKPT_DIR, exist_ok=True)

    # loading splits
    print("Loading PHOENIX keypoints...")
    train_raw = dict(np.load(TRAIN_KP, allow_pickle=True))
    dev_raw   = dict(np.load(DEV_KP,   allow_pickle=True))

    # building gloss from training set
    gloss_to_idx, idx_to_gloss = build_phoenix_vocab(train_raw)
    print(f"Gloss vocabulary: {len(gloss_to_idx)} tokens")

    # config
    cfg = MambaASLConfig()
    cfg.n_keypoints= N_KEYPOINTS
    cfg.vocab_size = len(gloss_to_idx)
    cfg.batch_size = batch_size
    cfg.max_epochs = max_epochs
    cfg.learning_rate = learning_rate
    cfg.grad_clip = grad_clip
    cfg.n_layers = n_layers

    # datasets
    train_ds = PhoenixDataset(TRAIN_KP, gloss_to_idx)
    dev_ds = PhoenixDataset(DEV_KP,   gloss_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    model = MambaASLFast(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max_epochs)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    best_val = float('inf')
    epochs_no_imp = 0

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input dim: {N_KEYPOINTS}, Vocab: {cfg.vocab_size}, LR: {learning_rate}")

    for epoch in range(1, max_epochs + 1):
        # training
        model.train()
        train_loss = 0.0
        for frames, frame_lens, labels, label_lens in tqdm(
                train_loader, desc=f"Epoch {epoch} train"):
            frames = frames.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda'):
                log_probs = model(frames)
                loss = ctc_loss(log_probs, labels, frame_lens, label_lens)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if any(torch.isnan(p).any() for p in model.parameters()):
                optimizer.zero_grad()
            else:
                train_loss += loss.item()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, frame_lens, labels, label_lens in tqdm(
                    dev_loader, desc=f"Epoch {epoch} val"):
                frames = frames.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda'):
                    log_probs = model(frames)
                    batch_loss = ctc_loss(log_probs, labels, frame_lens, label_lens)
                if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                    val_loss += batch_loss.item()
        val_loss /= len(dev_loader) if len(dev_loader) > 0 else 1
        scheduler.step()
        print(f"Epoch {epoch:3d}, train={train_loss:.4f}, val={val_loss:.4f}")
        # save best model
        ckpt = {
            'epoch': epoch, 'cfg': cfg,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'train_loss': train_loss, 'val_loss': val_loss,
            'gloss_to_idx': gloss_to_idx, 'idx_to_gloss': idx_to_gloss,
        }
        if val_loss < best_val:
            best_val      = val_loss
            epochs_no_imp = 0
            torch.save(ckpt, os.path.join(CKPT_DIR, 'phoenix_best.pt'))
            print(f"saved best checkpoint")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"early stopping at {epoch}, no improvement for {patience} epochs")
                break
    print(f"Best validation loss: {best_val:.4f}")
    return best_val

if __name__ == "__main__":
    train()
