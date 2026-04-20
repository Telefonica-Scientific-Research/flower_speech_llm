#!/usr/bin/env python3
"""
Evaluate a federated SpeechLLM checkpoint on MLS test sets.

Loads a checkpoint from FL training and evaluates on MLS test CSVs,
reporting WER per language and overall.

Usage:
  # Evaluate final FL model on all test languages:
  python evaluate_fl_model.py --checkpoint FL_SLAM_checkpoints/final_model.pt

  # Evaluate a specific round checkpoint:
  python evaluate_fl_model.py --checkpoint FL_SLAM_checkpoints/Checkpoint-round-200.ckpt

  # Evaluate on a specific language only:
  python evaluate_fl_model.py --checkpoint final_model.pt --test-dir fl_MLS_test --test-files test_german.csv

  # Custom model config:
  python evaluate_fl_model.py --checkpoint final_model.pt --llm-name TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
import sys
import argparse
import json
import re
from collections import defaultdict

import torch
import pytorch_lightning as pl
import pandas as pd
from jiwer import wer

from .trainer import SpeechLLMLightning
from .dataset import InstructionalAudioDataset, MyCollator

from torch.utils.data import DataLoader


def load_model(checkpoint_path, model_kwargs):
    """Instantiate SpeechLLMLightning and load checkpoint weights."""
    model = SpeechLLMLightning(**model_kwargs)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Handle both full state dicts and partial (LoRA-only) state dicts
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def evaluate_on_csv(model, csv_path, device="cuda", max_samples=None):
    """
    Evaluate model on a single test CSV and return per-sample results.
    """
    collator = MyCollator(
        audio_encoder_name=model.hparams.get("audio_encoder_name", "microsoft/wavlm-large"),
        tokenizer=model.llm_tokenizer,
    )

    dataset = InstructionalAudioDataset(
        csv_file=csv_path, mode="test", random_keys_prob=0.0
    )

    if max_samples and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collator, num_workers=2,
    )

    model = model.to(device)
    model.eval()

    results = {
        "wer_scores": [],
        "predictions": [],
        "targets": [],
    }

    print(f"  Evaluating {csv_path} ({len(loader)} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch

            if mel is not None:
                mel = mel.to(device)
            pre_tokenized_ids = pre_tokenized_ids.to(device)
            post_tokenized_ids = post_tokenized_ids.to(device)
            output_tokenized_ids = output_tokenized_ids.to(device)

            embeds, atts, label_ids = model.encode(
                mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
            )
            outputs = model(embeds, atts, label_ids)

            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()

            generated_text = model.llm_tokenizer.decode(
                predicted_ids[0], skip_special_tokens=False
            )
            target_text = model.llm_tokenizer.decode(
                output_tokenized_ids[0].cpu(), skip_special_tokens=False
            )

            extracted_pred = model.extract_prediction_values(generated_text)
            extracted_target = model.extract_prediction_values(target_text)

            # WER on transcript
            pred_transcript = extracted_pred.get("Transcript", "")
            target_transcript = extracted_target.get("Transcript", "")

            if target_transcript:
                wer_score = wer(target_transcript.lower(), pred_transcript.lower())
                results["wer_scores"].append(wer_score)
                results["predictions"].append(pred_transcript)
                results["targets"].append(target_transcript)

            if (batch_idx + 1) % 200 == 0:
                avg_wer = (
                    sum(results["wer_scores"]) / len(results["wer_scores"])
                    if results["wer_scores"]
                    else 0
                )
                print(f"    [{batch_idx+1}/{len(loader)}] Running WER: {avg_wer:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FL SpeechLLM checkpoint on MLS test sets"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    parser.add_argument(
        "--test-dir", default="fl_MLS_test",
        help="Directory containing test CSVs (default: fl_MLS_test)",
    )
    parser.add_argument(
        "--test-files", nargs="*", default=None,
        help="Specific test CSV filenames (default: all CSVs in test-dir)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per test file (default: all)",
    )
    parser.add_argument(
        "--output-json", default="eval_results.json",
        help="Output JSON file for results (default: eval_results.json)",
    )
    # Model config
    parser.add_argument("--audio-encoder-name", default="microsoft/wavlm-large")
    parser.add_argument("--llm-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--connector-name", default="linear")
    parser.add_argument("--audio-enc-dim", type=int, default=1024)
    parser.add_argument("--llm-dim", type=int, default=2048)
    parser.add_argument("--connector-k", type=int, default=2)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ---- Build model ----
    model_kwargs = {
        "audio_enc_dim": args.audio_enc_dim,
        "llm_dim": args.llm_dim,
        "audio_encoder_name": args.audio_encoder_name,
        "connector_name": args.connector_name,
        "llm_name": args.llm_name,
        "finetune_encoder": False,
        "connector_k": args.connector_k,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }

    print("=" * 70)
    print("MLS Federated Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Test dir:    {args.test_dir}")
    print(f"Device:      {args.device}")
    print(f"LLM:         {args.llm_name}")
    print(f"Encoder:     {args.audio_encoder_name}")
    print("=" * 70)

    model = load_model(args.checkpoint, model_kwargs)

    # ---- Find test CSVs ----
    if args.test_files:
        test_csvs = [os.path.join(args.test_dir, f) for f in args.test_files]
    else:
        if not os.path.isdir(args.test_dir):
            print(f"ERROR: Test directory not found: {args.test_dir}")
            sys.exit(1)
        test_csvs = sorted([
            os.path.join(args.test_dir, f)
            for f in os.listdir(args.test_dir)
            if f.endswith(".csv") and f != "test_all.csv"
        ])

    if not test_csvs:
        print("ERROR: No test CSV files found.")
        sys.exit(1)

    print(f"\nTest files: {len(test_csvs)}")
    for csv in test_csvs:
        print(f"  - {csv}")

    # ---- Evaluate ----
    all_results = {}
    all_wer_scores = []

    for csv_path in test_csvs:
        lang = os.path.basename(csv_path).replace("test_", "").replace(".csv", "")
        print(f"\n--- Evaluating: {lang} ---")

        results = evaluate_on_csv(
            model, csv_path,
            device=args.device,
            max_samples=args.max_samples,
        )

        if results["wer_scores"]:
            avg_wer = sum(results["wer_scores"]) / len(results["wer_scores"])
            all_wer_scores.extend(results["wer_scores"])
        else:
            avg_wer = float("nan")

        all_results[lang] = {
            "wer": avg_wer,
            "num_samples": len(results["wer_scores"]),
        }
        print(f"  {lang}: WER = {avg_wer:.4f}  ({len(results['wer_scores'])} samples)")

    # ---- Overall ----
    if all_wer_scores:
        overall_wer = sum(all_wer_scores) / len(all_wer_scores)
    else:
        overall_wer = float("nan")

    all_results["overall"] = {
        "wer": overall_wer,
        "num_samples": len(all_wer_scores),
    }

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Language':<15} {'WER':>8} {'Samples':>10}")
    print("-" * 35)
    for lang, res in sorted(all_results.items()):
        if lang != "overall":
            print(f"{lang:<15} {res['wer']:>8.4f} {res['num_samples']:>10d}")
    print("-" * 35)
    print(f"{'OVERALL':<15} {overall_wer:>8.4f} {len(all_wer_scores):>10d}")
    print("=" * 70)

    # ---- Save results ----
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
