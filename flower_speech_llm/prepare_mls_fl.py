#!/usr/bin/env python3
"""
Download Multilingual LibriSpeech (MLS) and create per-speaker FL client partitions.

Creates:
  fl_multilingual/       - Train CSVs (one per speaker across all languages)
  fl_MLS_dev_speaker/    - Dev CSVs (one per speaker across all languages)
  fl_MLS_test/           - Test CSVs (one per language + combined)

Each CSV has columns matching InstructionalAudioDataset:
  audio_path, transcript, gender, emotion, age, accent, isspeech

Usage:
  # Download all 7 non-English MLS languages (full train/dev/test):
  python prepare_mls_fl.py

  # Limit training samples per language (for faster experimentation):
  python prepare_mls_fl.py --max-train-per-lang 5000

  # Select specific languages:
  python prepare_mls_fl.py --languages german french spanish

  # Custom output base directory:
  python prepare_mls_fl.py --base-dir /path/to/data
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import pandas as pd
import soundfile as sf
from datasets import load_dataset, concatenate_datasets

# All MLS languages (including English)
ALL_LANGUAGES = [
    "english", "german", "dutch", "french", "spanish",
    "italian", "portuguese", "polish",
]

# English LibriSpeech split mapping (openslr/librispeech_asr)
# We combine clean + other for train; clean for dev/test
ENGLISH_SPLIT_MAP = {
    "train": [
        ("clean", "train.100"),
        ("clean", "train.360"),
        ("other", "train.500"),
    ],
    "dev": [
        ("clean", "validation"),
        ("other", "validation"),
    ],
    "test": [
        ("clean", "test"),
        ("other", "test"),
    ],
}


def _load_english_split(split_name):
    """Load English LibriSpeech from openslr/librispeech_asr as an iterable stream."""
    from itertools import chain

    configs_and_splits = ENGLISH_SPLIT_MAP.get(split_name)
    if not configs_and_splits:
        raise ValueError(f"Unknown split for English: {split_name}")

    iterables = []
    for config, hf_split in configs_and_splits:
        ds = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=hf_split,
            streaming=True,
            trust_remote_code=True,
        )
        iterables.append(ds)

    # Chain multiple streaming datasets together
    return chain(*iterables)


def _get_transcript(sample, lang):
    """Extract transcript from sample, handling field name differences."""
    if lang == "english":
        return sample.get("text", "").strip()
    return sample.get("transcript", "").strip()


def process_split(
    languages,
    split_name,
    audio_base_dir,
    out_dir,
    max_per_lang=None,
    partition_by_speaker=True,
):
    """
    Download one MLS split, save audio as FLAC, and write CSVs.

    If partition_by_speaker=True  → one CSV per speaker (for train/dev).
    If partition_by_speaker=False → one CSV per language + combined (for test).
    """
    os.makedirs(out_dir, exist_ok=True)

    # speaker_key -> list of row dicts
    speaker_data = defaultdict(list)
    lang_sample_counts = {}

    for lang in languages:
        print(f"\n--- [{lang}] Loading '{split_name}' split (streaming) ---")
        t0 = time.time()

        try:
            if lang == "english":
                ds = _load_english_split(split_name)
            else:
                ds = load_dataset(
                    "facebook/multilingual_librispeech",
                    lang,
                    split=split_name,
                    streaming=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            print(f"  [{lang}] ERROR loading dataset: {e}")
            continue

        audio_lang_dir = os.path.join(audio_base_dir, lang, split_name)
        os.makedirs(audio_lang_dir, exist_ok=True)

        count = 0
        skipped = 0

        for sample in ds:
            speaker_id = sample["speaker_id"]
            sample_id = sample["id"]
            audio_info = sample["audio"]

            # ---- Save audio as FLAC ----
            speaker_audio_dir = os.path.join(audio_lang_dir, str(speaker_id))
            os.makedirs(speaker_audio_dir, exist_ok=True)
            audio_path = os.path.join(speaker_audio_dir, f"{sample_id}.flac")

            if not os.path.exists(audio_path):
                try:
                    sf.write(
                        audio_path,
                        audio_info["array"],
                        audio_info["sampling_rate"],
                        format="FLAC",
                    )
                except Exception as e:
                    print(f"  [{lang}] WARNING: Failed to write {audio_path}: {e}")
                    skipped += 1
                    continue
            else:
                pass  # Resume: file already exists

            # ---- Build CSV row ----
            row = {
                "audio_path": os.path.abspath(audio_path),
                "transcript": _get_transcript(sample, lang),
                "gender": "",
                "emotion": "",
                "age": "",
                "accent": lang,
                "isspeech": True,
            }

            speaker_key = f"{lang}_{speaker_id}"
            speaker_data[speaker_key].append(row)

            count += 1
            if count % 2000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  [{lang}] {count:>8d} samples processed  ({rate:.0f} samples/sec)")

            if max_per_lang is not None and count >= max_per_lang:
                print(f"  [{lang}] Reached limit of {max_per_lang} samples.")
                break

        elapsed = time.time() - t0
        lang_sample_counts[lang] = count
        print(f"  [{lang}] Done: {count} samples, {skipped} skipped, {elapsed:.1f}s")

    # ---- Write CSV files ----
    if partition_by_speaker:
        # One CSV per speaker → fl_multilingual/client_0.csv, client_1.csv, ...
        sorted_speakers = sorted(speaker_data.keys())
        for i, speaker_key in enumerate(sorted_speakers):
            rows = speaker_data[speaker_key]
            df = pd.DataFrame(rows)
            csv_path = os.path.join(out_dir, f"client_{i}.csv")
            df.to_csv(csv_path, index=False)

        print(f"\n=> Created {len(sorted_speakers)} per-speaker CSVs in {out_dir}")
        return len(sorted_speakers)
    else:
        # Test: one CSV per language + one combined
        total = 0
        for lang in languages:
            lang_rows = []
            for key, rows in speaker_data.items():
                if key.startswith(f"{lang}_"):
                    lang_rows.extend(rows)
            if lang_rows:
                df = pd.DataFrame(lang_rows)
                csv_path = os.path.join(out_dir, f"test_{lang}.csv")
                df.to_csv(csv_path, index=False)
                print(f"  Test CSV [{lang}]: {len(lang_rows)} samples -> {csv_path}")
                total += len(lang_rows)

        # Combined test CSV
        all_rows = []
        for rows in speaker_data.values():
            all_rows.extend(rows)
        if all_rows:
            df = pd.DataFrame(all_rows)
            csv_path = os.path.join(out_dir, "test_all.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Combined test CSV: {len(all_rows)} samples -> {csv_path}")

        return total


def main():
    parser = argparse.ArgumentParser(
        description="Download MLS and create FL partitions for speech_llm_fl"
    )
    parser.add_argument(
        "--languages", nargs="+", default=ALL_LANGUAGES,
        help=f"MLS languages to download. Default: {ALL_LANGUAGES}",
    )
    parser.add_argument(
        "--base-dir", default=".",
        help="Base directory for output (default: current directory)",
    )
    parser.add_argument(
        "--audio-dir", default="mls_audio",
        help="Subdirectory for saved audio files (default: mls_audio)",
    )
    parser.add_argument(
        "--train-dir", default="fl_multilingual",
        help="Output directory for train CSVs (default: fl_multilingual)",
    )
    parser.add_argument(
        "--dev-dir", default="fl_MLS_dev_speaker",
        help="Output directory for dev CSVs (default: fl_MLS_dev_speaker)",
    )
    parser.add_argument(
        "--test-dir", default="fl_MLS_test",
        help="Output directory for test CSVs (default: fl_MLS_test)",
    )
    parser.add_argument(
        "--max-train-per-lang", type=int, default=None,
        help="Max training samples per language (default: all)",
    )
    parser.add_argument(
        "--max-dev-per-lang", type=int, default=None,
        help="Max dev samples per language (default: all)",
    )
    parser.add_argument(
        "--max-test-per-lang", type=int, default=None,
        help="Max test samples per language (default: all)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip train split processing",
    )
    parser.add_argument(
        "--skip-dev", action="store_true",
        help="Skip dev split processing",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Skip test split processing",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    audio_base = os.path.join(base, args.audio_dir)
    train_out = os.path.join(base, args.train_dir)
    dev_out = os.path.join(base, args.dev_dir)
    test_out = os.path.join(base, args.test_dir)

    print("=" * 70)
    print("MLS Federated Learning Data Preparation")
    print("=" * 70)
    print(f"Languages:  {args.languages}")
    print(f"Base dir:   {base}")
    print(f"Audio dir:  {audio_base}")
    print(f"Train dir:  {train_out}  (per-speaker CSVs)")
    print(f"Dev dir:    {dev_out}  (per-speaker CSVs)")
    print(f"Test dir:   {test_out}  (per-language CSVs)")
    if args.max_train_per_lang:
        print(f"Max train/lang: {args.max_train_per_lang}")
    if args.max_dev_per_lang:
        print(f"Max dev/lang:   {args.max_dev_per_lang}")
    if args.max_test_per_lang:
        print(f"Max test/lang:  {args.max_test_per_lang}")
    print("=" * 70)

    num_train_clients = 0
    num_dev_clients = 0

    # ---- Train ----
    if not args.skip_train:
        print("\n\n" + "=" * 70)
        print("TRAIN SPLIT")
        print("=" * 70)
        num_train_clients = process_split(
            languages=args.languages,
            split_name="train",
            audio_base_dir=audio_base,
            out_dir=train_out,
            max_per_lang=args.max_train_per_lang,
            partition_by_speaker=True,
        )

    # ---- Dev ----
    if not args.skip_dev:
        print("\n\n" + "=" * 70)
        print("DEV SPLIT")
        print("=" * 70)
        num_dev_clients = process_split(
            languages=args.languages,
            split_name="dev",
            audio_base_dir=audio_base,
            out_dir=dev_out,
            max_per_lang=args.max_dev_per_lang,
            partition_by_speaker=True,
        )

    # ---- Test ----
    if not args.skip_test:
        print("\n\n" + "=" * 70)
        print("TEST SPLIT")
        print("=" * 70)
        process_split(
            languages=args.languages,
            split_name="test",
            audio_base_dir=audio_base,
            out_dir=test_out,
            max_per_lang=args.max_test_per_lang,
            partition_by_speaker=False,
        )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if num_train_clients:
        print(f"Train clients (speakers): {num_train_clients}")
    if num_dev_clients:
        print(f"Dev clients (speakers):   {num_dev_clients}")

    # Count test files
    if os.path.exists(test_out):
        test_csvs = [f for f in os.listdir(test_out) if f.endswith(".csv")]
        print(f"Test CSV files:           {len(test_csvs)}")

    if num_train_clients:
        print(f"\n--- Update pyproject.toml ---")
        print(f'csv-train-dir = "./{args.train_dir}"')
        print(f'csv-dev-dir   = "./{args.dev_dir}"')
        print(f"")
        print(f"[tool.flwr.federations.local-simulation]")
        print(f"options.num-supernodes = {num_train_clients}")
    print("=" * 70)


if __name__ == "__main__":
    main()
