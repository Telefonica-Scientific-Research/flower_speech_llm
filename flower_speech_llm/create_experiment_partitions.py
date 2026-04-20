#!/usr/bin/env python3
"""
Create FL experiment partitions for MLS using Flower Datasets library.

Generates three experiment settings (all with 316 clients):
  A1: Multilingual-mixed IID clients (random assignment across all languages)
  B1: One-speaker-per-client (stratified by language, 316 speakers)
  B2: Same partition as B1 (different FL strategy — FedProx — handled at runtime)

Output directories:
  fl_A1_mixed_316/        — 316 CSVs for Setting A1
  fl_B1_speaker_316/      — 316 CSVs for Setting B1 (also used by B2)
  fl_dev_316/             — 316 dev CSVs (shared across experiments)

Usage:
  python create_experiment_partitions.py
  python create_experiment_partitions.py --num-clients 316 --base-dir .
  python create_experiment_partitions.py --seed 42
"""

import os
import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner


NUM_CLIENTS = 316
SEED = 42


def load_all_train_csvs(train_dir: str) -> pd.DataFrame:
    """Load all client_*.csv from train_dir into one DataFrame with a speaker_key column."""
    all_dfs = []
    csv_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".csv")])
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(train_dir, csv_file))
        # The original partition was by {lang}_{speaker_id}, extract from accent + file index
        # Each file = one speaker. Add speaker_key as the file stem
        speaker_key = csv_file.replace(".csv", "")
        df["speaker_key"] = speaker_key
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} CSVs → {len(combined):,} total samples")
    return combined


def create_a1_mixed_partition(pooled_df: pd.DataFrame, num_clients: int, out_dir: str, seed: int):
    """
    Setting A1: Multilingual-mixed IID clients.
    Pool all samples, randomly assign to num_clients clients using IidPartitioner.
    """
    print(f"\n{'='*70}")
    print(f"SETTING A1: Multilingual-mixed IID ({num_clients} clients)")
    print(f"{'='*70}")

    os.makedirs(out_dir, exist_ok=True)

    # Convert to HuggingFace Dataset for flwr-datasets
    # Drop audio-heavy columns — we only need to partition the metadata
    hf_dataset = Dataset.from_pandas(pooled_df.reset_index(drop=True))

    # Use IidPartitioner from Flower Datasets
    partitioner = IidPartitioner(num_partitions=num_clients)
    partitioner.dataset = hf_dataset

    lang_dist = Counter()
    total_samples = 0

    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        partition_df = partition.to_pandas()

        # Drop internal columns added by flwr-datasets
        for col in ["__index_level_0__"]:
            if col in partition_df.columns:
                partition_df = partition_df.drop(columns=[col])

        # Drop speaker_key (not needed in final CSV)
        csv_df = partition_df.drop(columns=["speaker_key"], errors="ignore")
        csv_path = os.path.join(out_dir, f"client_{i}.csv")
        csv_df.to_csv(csv_path, index=False)

        total_samples += len(csv_df)
        for lang in partition_df["accent"].values:
            lang_dist[lang] += 1

    print(f"Created {num_clients} CSVs in {out_dir}")
    print(f"Total samples: {total_samples:,}")
    print(f"Avg samples/client: {total_samples / num_clients:.0f}")
    print(f"Language distribution:")
    for lang, count in sorted(lang_dist.items()):
        print(f"  {lang}: {count:,} ({100*count/total_samples:.1f}%)")

    return out_dir


def create_b1_speaker_partition(
    pooled_df: pd.DataFrame, num_clients: int, out_dir: str, seed: int
):
    """
    Setting B1: One-speaker-per-client, stratified by language.
    Select num_clients speakers proportionally from each language, then
    use NaturalIdPartitioner(partition_by="speaker_key").
    """
    print(f"\n{'='*70}")
    print(f"SETTING B1: One-speaker-per-client ({num_clients} clients)")
    print(f"{'='*70}")

    os.makedirs(out_dir, exist_ok=True)

    # Count speakers per language
    speaker_lang = pooled_df.groupby("speaker_key")["accent"].first()
    lang_speakers = defaultdict(list)
    for spk, lang in speaker_lang.items():
        lang_speakers[lang].append(spk)

    total_speakers = sum(len(v) for v in lang_speakers.values())
    print(f"Total speakers available: {total_speakers}")

    # Stratified sampling: pick speakers proportionally per language
    rng = random.Random(seed)
    selected_speakers = []

    # Calculate proportional allocation
    allocations = {}
    remaining = num_clients
    sorted_langs = sorted(lang_speakers.keys(), key=lambda l: len(lang_speakers[l]))

    for i, lang in enumerate(sorted_langs):
        available = len(lang_speakers[lang])
        proportion = available / total_speakers
        if i == len(sorted_langs) - 1:
            # Last language gets the remainder
            alloc = remaining
        else:
            alloc = max(1, round(num_clients * proportion))
            alloc = min(alloc, available, remaining)
        allocations[lang] = alloc
        remaining -= alloc

    # If remaining > 0 due to rounding, distribute to languages with most available
    while remaining > 0:
        for lang in sorted(lang_speakers.keys(), key=lambda l: len(lang_speakers[l]), reverse=True):
            if remaining <= 0:
                break
            if allocations[lang] < len(lang_speakers[lang]):
                allocations[lang] += 1
                remaining -= 1

    print(f"\nSpeaker allocation per language:")
    for lang in sorted(allocations.keys()):
        avail = len(lang_speakers[lang])
        alloc = allocations[lang]
        print(f"  {lang}: {alloc} / {avail} speakers")

    # Sample speakers
    for lang, alloc in allocations.items():
        speakers = lang_speakers[lang]
        sampled = rng.sample(speakers, min(alloc, len(speakers)))
        selected_speakers.extend(sampled)

    print(f"\nTotal selected speakers: {len(selected_speakers)}")

    # Filter pooled_df to selected speakers only
    filtered_df = pooled_df[pooled_df["speaker_key"].isin(selected_speakers)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    print(f"Filtered samples: {len(filtered_df):,}")

    # Use NaturalIdPartitioner from Flower Datasets
    hf_dataset = Dataset.from_pandas(filtered_df)
    partitioner = NaturalIdPartitioner(partition_by="speaker_key")
    partitioner.dataset = hf_dataset

    actual_num = partitioner.num_partitions
    print(f"NaturalIdPartitioner created {actual_num} partitions")

    sample_counts = []
    lang_dist = Counter()

    for i in range(actual_num):
        partition = partitioner.load_partition(i)
        partition_df = partition.to_pandas()

        # Drop internal columns
        for col in ["__index_level_0__"]:
            if col in partition_df.columns:
                partition_df = partition_df.drop(columns=[col])

        csv_df = partition_df.drop(columns=["speaker_key"], errors="ignore")
        csv_path = os.path.join(out_dir, f"client_{i}.csv")
        csv_df.to_csv(csv_path, index=False)

        sample_counts.append(len(csv_df))
        for lang in partition_df["accent"].values:
            lang_dist[lang] += 1

    sample_counts = np.array(sample_counts)
    total = sample_counts.sum()

    print(f"\nCreated {actual_num} CSVs in {out_dir}")
    print(f"Total samples: {total:,}")
    print(f"Samples/client — Mean: {sample_counts.mean():.0f}, "
          f"Median: {np.median(sample_counts):.0f}, "
          f"Min: {sample_counts.min()}, Max: {sample_counts.max()}")
    print(f"Language distribution:")
    for lang, count in sorted(lang_dist.items()):
        print(f"  {lang}: {count:,} ({100*count/total:.1f}%)")

    # Save speaker mapping for reference
    mapping = partitioner.partition_id_to_natural_id
    mapping_path = os.path.join(out_dir, "partition_speaker_mapping.txt")
    with open(mapping_path, "w") as f:
        for pid, spk in sorted(mapping.items()):
            f.write(f"client_{pid}\t{spk}\n")
    print(f"Speaker mapping saved to {mapping_path}")

    return out_dir


def create_dev_partition(dev_dir: str, num_clients: int, out_dir: str, seed: int):
    """
    Create dev partitions: pool all dev CSVs and split IID into num_clients.
    Shared across all experiment settings.
    """
    print(f"\n{'='*70}")
    print(f"DEV PARTITION ({num_clients} clients)")
    print(f"{'='*70}")

    os.makedirs(out_dir, exist_ok=True)

    # Load all dev CSVs
    all_dfs = []
    csv_files = sorted([f for f in os.listdir(dev_dir) if f.endswith(".csv")])
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(dev_dir, csv_file))
        all_dfs.append(df)

    if not all_dfs:
        print(f"  WARNING: No dev CSVs found in {dev_dir}")
        return out_dir

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} dev CSVs → {len(combined):,} samples")

    # IID partition for dev
    hf_dataset = Dataset.from_pandas(combined)
    partitioner = IidPartitioner(num_partitions=num_clients)
    partitioner.dataset = hf_dataset

    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        partition_df = partition.to_pandas()
        for col in ["__index_level_0__"]:
            if col in partition_df.columns:
                partition_df = partition_df.drop(columns=[col])
        csv_path = os.path.join(out_dir, f"client_{i}.csv")
        partition_df.to_csv(csv_path, index=False)

    print(f"Created {num_clients} dev CSVs in {out_dir}")
    avg = len(combined) // num_clients
    print(f"Avg samples/client: {avg}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create FL experiment partitions for A1/B1/B2 using Flower Datasets"
    )
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--train-dir", default="fl_multilingual",
        help="Source train CSVs (speaker-partitioned, from prepare_mls_fl.py)",
    )
    parser.add_argument(
        "--dev-dir", default="fl_MLS_dev_speaker",
        help="Source dev CSVs (speaker-partitioned)",
    )
    parser.add_argument(
        "--a1-out", default="fl_A1_mixed_316",
        help="Output dir for Setting A1",
    )
    parser.add_argument(
        "--b1-out", default="fl_B1_speaker_316",
        help="Output dir for Setting B1 (and B2)",
    )
    parser.add_argument(
        "--dev-out", default="fl_dev_316",
        help="Output dir for shared dev partition",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    train_dir = os.path.join(base, args.train_dir)
    dev_dir = os.path.join(base, args.dev_dir)
    a1_out = os.path.join(base, args.a1_out)
    b1_out = os.path.join(base, args.b1_out)
    dev_out = os.path.join(base, args.dev_out)

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("MLS Federated Learning — Experiment Partition Generator")
    print("=" * 70)
    print(f"Source train dir: {train_dir}")
    print(f"Source dev dir:   {dev_dir}")
    print(f"Num clients:      {args.num_clients}")
    print(f"Seed:             {args.seed}")
    print(f"A1 output:        {a1_out}")
    print(f"B1/B2 output:     {b1_out}")
    print(f"Dev output:       {dev_out}")
    print("=" * 70)

    # ---- Load all train data ----
    pooled_df = load_all_train_csvs(train_dir)

    # ---- Setting A1: Mixed IID ----
    create_a1_mixed_partition(pooled_df, args.num_clients, a1_out, args.seed)

    # ---- Setting B1: One-speaker ----
    create_b1_speaker_partition(pooled_df, args.num_clients, b1_out, args.seed)

    # ---- Dev partition (shared) ----
    create_dev_partition(dev_dir, args.num_clients, dev_out, args.seed)

    # ---- Print run commands ----
    print(f"\n{'='*70}")
    print("EXPERIMENT RUN COMMANDS")
    print(f"{'='*70}")
    print()
    print("# Setting A1: Mixed-multilingual + FedAvg")
    print(f'flwr run . --run-config "csv-train-dir=./{args.a1_out} csv-dev-dir=./{args.dev_out}"')
    print()
    print("# Setting B1: One-speaker + FedAvg")
    print(f'flwr run . --run-config "csv-train-dir=./{args.b1_out} csv-dev-dir=./{args.dev_out}"')
    print()
    print("# Setting B2: One-speaker + FedProx (mu=0.01)")
    print(f'flwr run . --run-config "csv-train-dir=./{args.b1_out} csv-dev-dir=./{args.dev_out} strategy=fedprox fedprox-mu=0.01"')
    print()
    print("# Evaluate any checkpoint on test set:")
    print(f"python evaluate_fl_model.py --checkpoint FL_SLAM_checkpoints/final_model.pt --test-dir fl_MLS_test")
    print("=" * 70)


if __name__ == "__main__":
    main()
