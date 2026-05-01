import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEV_KP = "/content/Phoenix-2014T.dev"

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)

def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    hyp = list(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)

def ctc_greedy_decode(log_probs: torch.Tensor, idx_to_gloss: dict,
                      blank: int = 0) -> str:
    indices = log_probs.argmax(-1).tolist()
    collapsed = []
    prev = None
    for idx in indices:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    tokens = [i for i in collapsed if i != blank]
    return ' '.join(idx_to_gloss.get(i, '<unk>') for i in tokens)

def evaluate(ckpt_path: str, batch_size: int = 16):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_phoenix import PhoenixDataset, collate_fn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    gloss_to_idx = ckpt['gloss_to_idx']
    idx_to_gloss = ckpt['idx_to_gloss']
    print(f"Loaded epoch {ckpt['epoch']} | vocab={len(gloss_to_idx)}")

    from model.mamba_asl_fast import MambaASLFast
    model = MambaASLFast(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    dev_ds = PhoenixDataset(DEV_KP, gloss_to_idx)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    total_wer, total_cer, n = 0.0, 0.0, 0
    total_inference_seconds = 0.0
    sample_idx = 0
    with torch.no_grad():
        for frames, frame_lens, labels, label_lens in tqdm(dev_loader, desc="Evaluating"):
            frames = frames.to(device)
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda'):
                log_probs = model(frames)
            total_inference_seconds += time.perf_counter() - t0
            label_offset = 0
            for b in range(frames.size(0)):
                T = frame_lens[b].item()
                seq_log_probs = log_probs[:T, b, :]
                hypothesis = ctc_greedy_decode(seq_log_probs, idx_to_gloss)
                L = label_lens[b].item()
                ref_tokens = labels[label_offset:label_offset + L].tolist()
                reference = ' '.join(idx_to_gloss.get(i, '<unk>') for i in ref_tokens)
                label_offset += L
                total_wer += wer(reference, hypothesis)
                total_cer += cer(reference, hypothesis)
                n += 1
                if sample_idx < 5:
                    print(f"Sample: {sample_idx + 1}")
                    print(f"REF: {reference}")
                    print(f"HYP: {hypothesis}")
                sample_idx += 1
    avg_wer = total_wer / n
    avg_cer = total_cer / n
    print(f"\nDev samples: {n}")
    print(f"WER: {avg_wer:.4f}  ({avg_wer*100:.2f}%)")
    print(f"CER: {avg_cer:.4f}  ({avg_cer*100:.2f}%)")
    print(f"Total inference time: {total_inference_seconds:.4f}s")
    print(f"Avg time per clip: {total_inference_seconds/n:.6f}s")
    return avg_wer, avg_cer

if __name__ == "__main__":
    evaluate(sys.argv[1])