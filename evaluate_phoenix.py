import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Transformer_ASL import TransformerASL
from train_phoenix import DEV_KP
from train_phoenix import PhoenixDataset
from train_phoenix import collate_phoenix_batch
from train_phoenix import ctc_greedy_tokens
from train_phoenix import make_padding_mask
from train_phoenix import tokens_to_gloss


def edit_distance(reference_words: list, predicted_words: list) -> int:
    previous_row = list(range(len(predicted_words) + 1))

    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]

        for predicted_index, predicted_word in enumerate(predicted_words, start=1):
            insert_cost = current_row[predicted_index - 1] + 1
            delete_cost = previous_row[predicted_index] + 1
            replace_cost = previous_row[predicted_index - 1]

            if reference_word != predicted_word:
                replace_cost += 1

            current_row.append(min(insert_cost, delete_cost, replace_cost))
            
        previous_row = current_row

    return previous_row[-1]


def run_phoenix_evaluation(checkpoint_path: str, batch_size: int = 16) -> dict:
    # cuda yay!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint["cfg"]
    gloss_to_index = checkpoint["gloss_to_idx"]
    index_to_gloss = checkpoint["idx_to_gloss"]

    model = TransformerASL(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dev_dataset = PhoenixDataset(DEV_KP, gloss_to_index)

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_phoenix_batch,
        num_workers=4,
    )

    total_word_errors = 0
    total_reference_words = 0
    total_clips = 0
    total_inference_seconds = 0.0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating PHOENIX dev"):
            (
                padded_keypoints,
                keypoint_lengths,
                gloss_labels,
                gloss_lengths,
            ) = batch

            padded_keypoints = padded_keypoints.to(device)
            keypoint_lengths = keypoint_lengths.to(device)

            padding_mask = make_padding_mask(padded_keypoints, keypoint_lengths)

            start_seconds = time.perf_counter()
            gloss_log_probs = model(padded_keypoints, src_key_padding_mask=padding_mask)

            end_seconds = time.perf_counter()
            total_inference_seconds += end_seconds - start_seconds

            label_start = 0

            for batch_index in range(padded_keypoints.size(0)):
                clip_frame_count = keypoint_lengths[batch_index].item()
                predicted_tokens = ctc_greedy_tokens(gloss_log_probs[:clip_frame_count, batch_index, :])
                predicted_gloss = tokens_to_gloss(predicted_tokens, index_to_gloss)

                label_count = gloss_lengths[batch_index].item()
                label_end = label_start + label_count
                target_tokens = gloss_labels[label_start:label_end].tolist()
                reference_gloss = tokens_to_gloss(target_tokens, index_to_gloss)
                label_start = label_end

                reference_words = reference_gloss.split()
                predicted_words = predicted_gloss.split()

                total_word_errors += edit_distance(reference_words, predicted_words)
                total_reference_words += len(reference_words)
                total_clips += 1

    word_error_rate = total_word_errors / max(total_reference_words, 1)
    seconds_per_clip = total_inference_seconds / max(total_clips, 1)

    results = {
        "device": str(device),
        "word_error_rate": word_error_rate,
        "total_inference_seconds": total_inference_seconds,
        "seconds_per_clip": seconds_per_clip,
    }

    print(f"Device: {results['device']}")
    print(f"Word Error Rate: {results['word_error_rate']:.4f} ({results['word_error_rate'] * 100:.2f}%)")
    print(f"Total Inference Time: {results['total_inference_seconds']:.4f} seconds")
    print(f"Average Inference Time Per Clip: {results['seconds_per_clip']:.6f} seconds")

    return results

if __name__ == "__main__":
    run_phoenix_evaluation(sys.argv[1])