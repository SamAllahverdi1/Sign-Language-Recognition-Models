import sys
import time
import torch

from jiwer import wer
from torch.utils.data import DataLoader
from tqdm import tqdm

from LSTM_ASL import ConvBiLSTMCTC

from train_phoenix import DEV_KP
from train_phoenix import PhoenixDataset
from train_phoenix import collate_phoenix_batch
from train_phoenix import ctc_greedy_tokens
from train_phoenix import tokens_to_text

def run_phoenix_evaluation(checkpoint_path: str, batch_size: int = 16) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint["cfg"]

    char_to_index = checkpoint["char_to_idx"]
    index_to_char = checkpoint["idx_to_char"]
    
    feature_mean = checkpoint["feature_mean"].cpu()
    feature_std = checkpoint["feature_std"].cpu()

    model = ConvBiLSTMCTC(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dev_dataset = PhoenixDataset(DEV_KP, char_to_index, feature_mean, feature_std)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_phoenix_batch, num_workers=2, pin_memory=True)

    all_predictions = []
    all_references = []

    with torch.no_grad():

        for batch in tqdm(dev_loader, desc="Evaluating PHOENIX dev"):

            (padded_keypoints, keypoint_lengths, text_labels, text_lengths) = batch

            padded_keypoints = padded_keypoints.to(device)
            text_log_probs = model(padded_keypoints)

            label_start = 0

            for batch_index in range(padded_keypoints.size(0)):

                clip_frame_count = keypoint_lengths[batch_index].item()

                predicted_tokens = ctc_greedy_tokens(text_log_probs[:clip_frame_count, batch_index, :])
                predicted_text = tokens_to_text(predicted_tokens, index_to_char)

                label_count = text_lengths[batch_index].item()
                label_end = label_start + label_count

                target_tokens = text_labels[label_start:label_end].tolist()
                reference_text = tokens_to_text(target_tokens, index_to_char)

                label_start = label_end

                all_predictions.append(predicted_text)
                all_references.append(reference_text)

    word_error_rate = wer(all_references, all_predictions)

    sample_keypoints, sample_lengths, _, _ = next(iter(dev_loader))
    sample_keypoints = sample_keypoints[:1].to(device)
    num_frames = sample_lengths[0].item()

    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_keypoints)

    if device.type == "cuda":
        torch.cuda.synchronize()

    n_runs = 100
    forward_times_ms = []

    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_seconds = time.perf_counter()
            _ = model(sample_keypoints)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_times_ms.append((time.perf_counter() - start_seconds) * 1000)

    avg_ms = sum(forward_times_ms) / len(forward_times_ms)
    std_ms = (sum((t - avg_ms) ** 2 for t in forward_times_ms) / len(forward_times_ms)) ** 0.5
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "device": str(device),
        "word_error_rate": word_error_rate,
        "parameters": param_count,
        "seq_latency_ms": avg_ms,
        "seq_latency_std_ms": std_ms,
        "per_frame_latency_ms": avg_ms / num_frames,
    }

    print(f"WER: {results['word_error_rate']}")
    print(f"Parameters: {results['parameters']}")
    print(f"Per-frame latency: {results['per_frame_latency_ms']} ms")
    print(f"Seq latency: {results['seq_latency_ms']} +- {results['seq_latency_std_ms']} ms")

    for sample_index in range(min(3, len(all_predictions))):
        print(f"\nRef: {all_references[sample_index]}")
        print(f"Pred: {all_predictions[sample_index]}")

    return results

if __name__ == "__main__":
    run_phoenix_evaluation(sys.argv[1])