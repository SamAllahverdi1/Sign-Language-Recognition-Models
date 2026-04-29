import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Transformer_ASL import TransformerASL
from train_phoenix import DEV_KP, PhoenixDataset, collate_fn, ctc_greedy_tokens, tokens_to_gloss


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            current = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = current
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def evaluate(ckpt_path: str, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    gloss_to_idx = checkpoint["gloss_to_idx"]
    idx_to_gloss = checkpoint["idx_to_gloss"]

    model = TransformerASL(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dev_ds = PhoenixDataset(DEV_KP, gloss_to_idx)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    total_wer = 0.0
    total_samples = 0
    total_inference_time = 0.0
    sample_idx = 0

    with torch.no_grad():
        for frames, frame_lens, labels, label_lens in tqdm(dev_loader, desc="Evaluating"):
            frames = frames.to(device)
            frame_lens = frame_lens.to(device)

            max_len = frames.size(1)
            padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= frame_lens.unsqueeze(1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Removed autocast to disable AMP per previous instructions
            log_probs = model(frames, src_key_padding_mask=padding_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_inference_time += (end_time - start_time)

            label_offset = 0
            for batch_idx in range(frames.size(0)):
                seq_len = frame_lens[batch_idx].item()
                pred_tokens = ctc_greedy_tokens(log_probs[:seq_len, batch_idx, :])
                hypothesis = tokens_to_gloss(pred_tokens, idx_to_gloss)

                target_len = label_lens[batch_idx].item()
                target_tokens = labels[label_offset:label_offset + target_len].tolist()
                reference = tokens_to_gloss(target_tokens, idx_to_gloss)
                label_offset += target_len

                total_wer += wer(reference, hypothesis)
                total_samples += 1

                if sample_idx < 5:
                    print(f"\n--- Sample {sample_idx + 1} ---")
                    print(f"REF: {reference}")
                    print(f"HYP: {hypothesis}")
                sample_idx += 1

    avg_wer = total_wer / total_samples
    average_time = total_inference_time / total_samples
    avg_batch_time = total_inference_time / len(dev_loader)
    print(f"\nDev samples : {total_samples}")
    print(f"WER         : {avg_wer:.4f} ({avg_wer * 100:.2f}%)")
    print(f"Avg Sample Inference Time: {average_time:.4f}s")
    print(f"Avg Batch Time: {avg_batch_time:.4f}s")
    return avg_wer


if __name__ == "__main__":
    evaluate(sys.argv[1])
