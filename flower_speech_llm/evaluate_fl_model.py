#!/usr/bin/env python3
"""
Evaluate a federated SpeechLLM checkpoint on MLS test sets.

Supports both model types:
  - speech-llm  (WavLM + connector + TinyLlama)
  - voxtral     (VoxtralForConditionalGeneration end-to-end)

For Voxtral, evaluation does NOT require knowing the language of the test
audio — the prompt uses a generic "transcribe" request and the model must
infer the language from the audio content.

Usage:
  # Speech-LLM (default):
  python evaluate_fl_model.py --checkpoint FL_SLAM_checkpoints/final_model.pt

  # Voxtral:
  python evaluate_fl_model.py --model-type voxtral \
      --checkpoint FL_SLAM_checkpoints/final_model.pt

  # Evaluate a specific round checkpoint:
  python evaluate_fl_model.py --checkpoint FL_SLAM_checkpoints/Checkpoint-round-200.ckpt

  # Evaluate on a specific language only:
  python evaluate_fl_model.py --checkpoint final_model.pt --test-dir fl_MLS_test --test-files test_german.csv
"""

import os
import sys
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
import pandas as pd
import torchaudio
from jiwer import wer

from .trainer import SpeechLLMLightning
from .dataset import InstructionalAudioDataset, MyCollator
from .model.voxtral import get_voxtral
from .trainer_voxtral import VoxtralLightning

from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Speech-LLM (WavLM) model loading + evaluation
# ---------------------------------------------------------------------------

def load_speech_llm(checkpoint_path, model_kwargs):
    """Instantiate SpeechLLMLightning and load checkpoint weights.

    Works with both adapter-only checkpoints (LoRA + connector, ~90 keys)
    and legacy full state_dicts (~779 keys).  Frozen base weights are
    always loaded from pretrained HuggingFace models.
    """
    model = SpeechLLMLightning(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    n_loaded = sum(1 for k in state_dict if k in dict(model.named_parameters()))
    print(f"Loaded speech-llm checkpoint: {checkpoint_path} ({len(state_dict)} keys, "
          f"{n_loaded} matched)")
    return model


def _extract_field_robust(text, field_name):
    """Extract a field value from a (possibly malformed) JSON-like string.

    Teacher-forced argmax decoding often produces slightly garbled JSON
    (misaligned quotes, missing delimiters) that breaks ``json.loads``.
    This helper uses a regex to pull out the value for *field_name*
    regardless of surrounding JSON validity.
    """
    pattern = rf'"{field_name}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def evaluate_speech_llm_on_csv(model, csv_path, device="cuda", max_samples=None):
    """Evaluate WavLM+LLM model on a single test CSV."""
    collator = MyCollator(
        audio_encoder_name=model.hparams.get("audio_encoder_name", "microsoft/wavlm-large"),
        tokenizer=model.llm_tokenizer,
    )
    dataset = InstructionalAudioDataset(csv_file=csv_path, mode="test", random_keys_prob=0.0)

    if max_samples and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=2)

    model = model.to(device)
    model.eval()
    results = {"wer_scores": [], "predictions": [], "targets": []}

    print(f"  Evaluating {csv_path} ({len(loader)} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask = batch

            if mel is not None:
                mel = mel.to(device)
                mel_mask = mel_mask.to(device)
            pre_ids = pre_ids.to(device)
            pre_mask = pre_mask.to(device)
            post_ids = post_ids.to(device)
            post_mask = post_mask.to(device)
            out_ids = out_ids.to(device)
            out_mask = out_mask.to(device)

            embeds, atts, label_ids = model.encode(
                mel, mel_mask, pre_ids, pre_mask, post_ids, post_mask, out_ids, out_mask
            )
            outputs = model(embeds, atts, label_ids)

            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()

            # Decode only output-portion predictions (where labels != -100).
            # This mirrors the Voxtral evaluation and avoids garbage from
            # input/speech positions that breaks JSON extraction.
            label_mask = (label_ids[0] != -100).cpu()
            output_pred_ids = predicted_ids[0][label_mask]
            generated_text = model.llm_tokenizer.decode(output_pred_ids, skip_special_tokens=True)

            target_text = model.llm_tokenizer.decode(
                out_ids[0].cpu(), skip_special_tokens=True
            )

            # Robust field extraction — tolerates slightly malformed JSON
            # produced by teacher-forced argmax decoding.
            pred_transcript = _extract_field_robust(generated_text, "Transcript")
            target_transcript = _extract_field_robust(target_text, "Transcript")

            if target_transcript:
                wer_score = wer(target_transcript.lower(), pred_transcript.lower())
                results["wer_scores"].append(wer_score)
                results["predictions"].append(pred_transcript)
                results["targets"].append(target_transcript)

            if (batch_idx + 1) % 200 == 0:
                avg = sum(results["wer_scores"]) / len(results["wer_scores"]) if results["wer_scores"] else 0
                print(f"    [{batch_idx+1}/{len(loader)}] Running WER: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# Voxtral model loading + evaluation
# ---------------------------------------------------------------------------

def load_voxtral_model(checkpoint_path, args):
    """Instantiate VoxtralLightning and load checkpoint weights."""
    processor, model = get_voxtral(
        model_name=args.voxtral_model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        finetune_encoder=False,
        cache_dir=args.model_cache_dir,
    )
    lightning_model = VoxtralLightning(model=model, processor=processor)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    lightning_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded voxtral checkpoint: {checkpoint_path}")
    return lightning_model


class VoxtralEvalDataset(Dataset):
    """Reads a test CSV with columns: audio_path, transcript.

    Returns raw waveform + text for evaluation. No language info required.
    """

    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["audio_path"])
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        return {
            "audio": {"array": waveform.squeeze(0).numpy(), "sampling_rate": 16000},
            "text": str(row.get("transcript", row.get("text", ""))),
        }


class VoxtralEvalCollator:
    """Builds Voxtral evaluation batches WITHOUT language information.

    Uses a generic transcription prompt so the model must infer the
    language from the audio content.
    """

    def __init__(self, processor, model_id: str):
        self.processor = processor
        self.model_id = model_id

    def __call__(self, features):
        texts = [f["text"] for f in features]
        audios = [f["audio"]["array"] for f in features]

        # Use a generic prompt — no language hint.
        # "en" is used as a placeholder for the prompt template; the model
        # transcribes whatever language it hears regardless.
        prompt = self.processor.apply_transcription_request(
            language="en",
            model_id=self.model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )

        passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}

        tok = self.processor.tokenizer
        prompt_ids = prompt["input_ids"]
        prompt_attn = prompt["attention_mask"]

        text_tok = tok(texts, add_special_tokens=False, padding=False,
                       truncation=True, max_length=256, return_tensors=None)

        input_ids_list, attn_list, labels_list = [], [], []
        for i, t_ids in enumerate(text_tok["input_ids"]):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            ids = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]
            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids_list)

        def _pad(seq, fill):
            return seq + [fill] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor([_pad(x, pad_id) for x in input_ids_list], dtype=torch.long),
            "attention_mask": torch.tensor([_pad(x, 0) for x in attn_list], dtype=torch.long),
            "labels": torch.tensor([_pad(x, -100) for x in labels_list], dtype=torch.long),
        }
        for k, v in passthrough.items():
            batch[k] = v

        # Keep ground-truth texts for WER computation
        batch["_ground_truth_texts"] = texts
        return batch


def evaluate_voxtral_on_csv(model, csv_path, device="cuda", max_samples=None,
                            voxtral_model_name="mistralai/Voxtral-Mini-3B-2507"):
    """Evaluate Voxtral model on a single test CSV (language-agnostic)."""
    collator = VoxtralEvalCollator(model.processor, voxtral_model_name)
    dataset = VoxtralEvalDataset(csv_path)

    if max_samples and max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    model = model.to(device)
    model.eval()
    results = {"wer_scores": [], "predictions": [], "targets": []}

    print(f"  Evaluating {csv_path} ({len(loader)} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            ground_truths = batch.pop("_ground_truth_texts")
            labels = batch["labels"]

            # Move tensors to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model.model(**batch)
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()

            # Decode only the label portion (where labels != -100)
            for i in range(labels.size(0)):
                label_mask = labels[i] != -100
                pred_ids = predicted_ids[i][label_mask]
                pred_text = model.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                target_text = ground_truths[i].strip()

                if target_text:
                    wer_score = wer(target_text.lower(), pred_text.lower())
                    results["wer_scores"].append(wer_score)
                    results["predictions"].append(pred_text)
                    results["targets"].append(target_text)

            if (batch_idx + 1) % 200 == 0:
                avg = sum(results["wer_scores"]) / len(results["wer_scores"]) if results["wer_scores"] else 0
                print(f"    [{batch_idx+1}/{len(loader)}] Running WER: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FL SpeechLLM checkpoint on MLS test sets"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--test-dir", default="fl_MLS_test", help="Directory containing test CSVs")
    parser.add_argument("--test-files", nargs="*", default=None, help="Specific test CSV filenames")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per test file")
    parser.add_argument("--output-json", default="eval_results.json", help="Output JSON file for results")

    # Model selection
    parser.add_argument("--model-type", default="speech-llm",
                        choices=["speech-llm", "voxtral"],
                        help="Model type: speech-llm (WavLM+TinyLlama) or voxtral")

    # Speech-LLM config
    parser.add_argument("--audio-encoder-name", default="microsoft/wavlm-large")
    parser.add_argument("--llm-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--connector-name", default="linear")
    parser.add_argument("--audio-enc-dim", type=int, default=1024)
    parser.add_argument("--llm-dim", type=int, default=2048)
    parser.add_argument("--connector-k", type=int, default=2)

    # Shared config
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # Voxtral config
    parser.add_argument("--voxtral-model-name", default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--model-cache-dir", default="")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print("MLS Federated Model Evaluation")
    print("=" * 70)
    print(f"Model type:  {args.model_type}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Test dir:    {args.test_dir}")
    print(f"Device:      {args.device}")
    if args.model_type == "voxtral":
        print(f"Model:       {args.voxtral_model_name}")
        print(f"Language:    unknown (language-agnostic evaluation)")
    else:
        print(f"LLM:         {args.llm_name}")
        print(f"Encoder:     {args.audio_encoder_name}")
    print("=" * 70)

    # ---- Build model ----
    if args.model_type == "voxtral":
        model = load_voxtral_model(args.checkpoint, args)
    else:
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
        model = load_speech_llm(args.checkpoint, model_kwargs)

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

        if args.model_type == "voxtral":
            results = evaluate_voxtral_on_csv(
                model, csv_path, device=args.device,
                max_samples=args.max_samples,
                voxtral_model_name=args.voxtral_model_name,
            )
        else:
            results = evaluate_speech_llm_on_csv(
                model, csv_path, device=args.device,
                max_samples=args.max_samples,
            )

        if results["wer_scores"]:
            avg_wer = sum(results["wer_scores"]) / len(results["wer_scores"])
            all_wer_scores.extend(results["wer_scores"])
        else:
            avg_wer = float("nan")

        all_results[lang] = {"wer": avg_wer, "num_samples": len(results["wer_scores"])}
        print(f"  {lang}: WER = {avg_wer:.4f}  ({len(results['wer_scores'])} samples)")

    # ---- Overall ----
    overall_wer = sum(all_wer_scores) / len(all_wer_scores) if all_wer_scores else float("nan")
    all_results["overall"] = {"wer": overall_wer, "num_samples": len(all_wer_scores)}

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
