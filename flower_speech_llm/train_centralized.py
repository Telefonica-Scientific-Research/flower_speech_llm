#!/usr/bin/env python3
"""
Centralized (non-federated) training of the SpeechLLM model.

This script trains the same WavLM + TinyLlama pipeline used in the FL
experiments, but pools all client data into a single centralized dataset.
Training hyperparameters match the FL configuration from pyproject.toml
so results are directly comparable.

Usage:
  # Default — matches FL config (40-round equivalent):
  python -m flower_speech_llm.train_centralized

  # Resume from FL checkpoint:
  python -m flower_speech_llm.train_centralized \
      --pretrained-checkpoint FL_SLAM_checkpoints/final_model.pt

  # Quick test run:
  python -m flower_speech_llm.train_centralized --max-epochs 1 --limit-train-batches 50

  # Use specific GPUs:
  python -m flower_speech_llm.train_centralized --devices 0,1

  # With wandb logging:
  python -m flower_speech_llm.train_centralized --wandb-project speech-llm-centralized
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

from .trainer import SpeechLLMLightning, save_trainable_state_dict, load_trainable_state_dict
from .dataset import InstructionalAudioDataset, MyCollator


def _build_run_dir(args):
    """Create a unique run sub-directory under args.output_dir.

    Layout: <output-dir>/<encoder>_<llm>_lr<r>a<alpha>_enc<0|1>_llm<0|1>_bs<B>x<A>_<YYYYMMDD_HHMMSS>/
    Copies the config YAML into the run directory for reproducibility.
    """
    enc_short = args.audio_encoder_name.split("/")[-1].lower()
    llm_short = args.llm_name.split("/")[-1].lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (
        f"{enc_short}_{llm_short}"
        f"_lr{args.lora_r}a{args.lora_alpha}"
        f"_enc{int(args.finetune_encoder)}_llm{int(args.finetune_llm)}"
        f"_bs{args.train_batch_size}x{args.grad_accumulate_steps}"
        f"_{ts}"
    )
    run_dir = os.path.join(args.output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    # Copy config YAML for reproducibility
    if args.config and os.path.exists(args.config):
        shutil.copy2(args.config, os.path.join(run_dir, os.path.basename(args.config)))

    args.output_dir = run_dir
    return run_dir


class LogProgressBar(TQDMProgressBar):
    """Progress bar that prints newlines instead of \\r for SLURM log files."""

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 120
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 120
        bar.ascii = True
        return bar

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx % self.refresh_rate == 0:
            print(self.train_progress_bar, flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if batch_idx % self.refresh_rate == 0:
            print(self.val_progress_bar, flush=True)


class AdapterCheckpoint(pl.Callback):
    """Save adapter-only .pt files after each validation run.

    These lightweight checkpoints (~12 MB) can be directly loaded by
    ``evaluate_fl_model.py`` without extracting state_dict from PL
    checkpoint format.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        step = trainer.global_step
        wer = trainer.callback_metrics.get("val/wer")
        tag = f"epoch{epoch:02d}-step{step}"
        if wer is not None:
            tag += f"-wer{wer:.4f}"
        path = os.path.join(self.output_dir, f"adapter-{tag}.pt")
        save_trainable_state_dict(pl_module, path)
        print(f"  💾 Adapter checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Comparable training budget calculation:
#
# FL setup (pyproject.toml):
#   40 rounds × fraction_fit=0.3 × 316 supernodes = ~95 clients/round
#   Each client: 10 local_epochs × 200 batches/epoch = 2000 batches
#   batch_size=4, grad_accumulate=4 → effective batch = 16
#   Total samples seen per round: 95 clients × 2000 × 4 = 760,000
#   Total samples across all rounds: 40 × 760,000 = 30,400,000
#
# Centralized equivalent:
#   With batch_size=4, grad_accum=4, effective_batch=16
#   30,400,000 / 4 = 7,600,000 batches = 7,600,000 / 200 ≈ 38,000 "epochs"
#   But the dataset has ~1.7M samples, so 30.4M / 1.7M ≈ 18 epochs
#   We use max_epochs=20 and limit_train_batches=None (full dataset) as default.
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Centralized training of SpeechLLM (WavLM + TinyLlama)"
    )

    # ---- Config file ----
    p.add_argument("--config", default="",
                   help="Path to a YAML config file. CLI args override config values.")

    # ---- Model config (matches pyproject.toml) ----
    p.add_argument("--audio-enc-dim", type=int, default=1024)
    p.add_argument("--llm-dim", type=int, default=2048)
    p.add_argument("--audio-encoder-name", default="microsoft/wavlm-large")
    p.add_argument("--connector-name", default="linear")
    p.add_argument("--llm-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--finetune-encoder", action="store_true", default=False)
    p.add_argument("--finetune-llm", action="store_true", default=True)
    p.add_argument("--no-finetune-llm", dest="finetune_llm", action="store_false")
    p.add_argument("--connector-k", type=int, default=2)
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)

    # ---- Optimizer config (matches pyproject.toml) ----
    p.add_argument("--max-lr", type=float, default=1e-4)
    p.add_argument("--total-training-step", type=int, default=2_000_000)
    p.add_argument("--warmup-steps", type=int, default=1000)

    # ---- Training config ----
    p.add_argument("--max-epochs", type=int, default=20,
                   help="Number of full passes over the pooled dataset (default: 20, "
                        "comparable to 40 FL rounds × 0.3 fraction × 10 local epochs)")
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--grad-accumulate-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size × accum)")
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--limit-train-batches", type=int, default=None,
                   help="Limit training batches per epoch (default: None = full dataset)")
    p.add_argument("--limit-val-batches", type=int, default=200,
                   help="Limit validation batches per epoch")
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--val-check-interval", type=float, default=0.25,
                   help="Run validation every N fraction of an epoch (default: 0.25)")

    # ---- Data config ----
    p.add_argument("--csv-train-dir",
                   default="flower_speech_llm/fl_A1_mixed_316",
                   help="Directory with training CSV files (all pooled together)")
    p.add_argument("--csv-dev-dir",
                   default="flower_speech_llm/fl_MLS_dev_speaker",
                   help="Directory with dev/validation CSV files")

    # ---- Checkpoint / resume ----
    p.add_argument("--pretrained-checkpoint", default="",
                   help="Path to a .pt/.ckpt to initialize weights from")
    p.add_argument("--output-dir", default="centralized_checkpoints",
                   help="Directory to save checkpoints")

    # ---- Hardware ----
    p.add_argument("--devices", default="auto",
                   help="GPU devices: 'auto', number, or comma-separated ids (e.g. '0,1')")
    p.add_argument("--strategy", default="auto",
                   help="PL strategy: 'auto', 'ddp', 'deepspeed_stage_2', etc.")
    p.add_argument("--precision", default="bf16-mixed",
                   help="Training precision (default: bf16-mixed)")

    # ---- Logging ----
    p.add_argument("--wandb-project", default="",
                   help="W&B project name (empty = no wandb)")
    p.add_argument("--run-name", default="centralized-speech-llm",
                   help="W&B run name / checkpoint prefix")
    p.add_argument("--log-every-n-steps", type=int, default=50,
                   help="Print progress bar every N batches (default: 50)")

    # ---- Two-pass parse: load YAML defaults, then let CLI override ----
    # First pass: just get --config
    preliminary, _ = p.parse_known_args()
    if preliminary.config and os.path.exists(preliminary.config):
        with open(preliminary.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # Map YAML keys (kebab-case) → argparse dest (underscore)
        defaults = {k.replace("-", "_"): v for k, v in yaml_cfg.items()}
        # Handle YAML null → Python None
        for k, v in defaults.items():
            if v is None:
                defaults[k] = None
        p.set_defaults(**defaults)
        print(f"Loaded config from: {preliminary.config}")

    return p.parse_args()


def build_pooled_dataloader(csv_dir, collator, batch_size, num_workers, shuffle, mode="train"):
    """Load all CSV files in csv_dir, merge into one ConcatDataset, return a single DataLoader."""
    csv_files = sorted(
        os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in: {csv_dir}")

    random_keys_prob = 0.1 if mode == "train" else 0.0
    datasets = [
        InstructionalAudioDataset(csv_file=f, mode=mode, random_keys_prob=random_keys_prob)
        for f in csv_files
    ]
    combined = ConcatDataset(datasets)
    print(f"✅ Pooled {len(csv_files)} CSVs → {len(combined)} samples ({mode})")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("medium")

    # ---- Create unique run directory ----
    run_dir = _build_run_dir(args)

    # ---- Build model ----
    print("=" * 70)
    print("Centralized SpeechLLM Training")
    print("=" * 70)
    print(f"LLM:          {args.llm_name}")
    print(f"Encoder:      {args.audio_encoder_name}")
    print(f"LoRA:         r={args.lora_r}, alpha={args.lora_alpha}" if args.use_lora else "LoRA: off")
    print(f"Train dir:    {args.csv_train_dir}")
    print(f"Dev dir:      {args.csv_dev_dir}")
    print(f"Max epochs:   {args.max_epochs}")
    print(f"Batch size:   {args.train_batch_size} × {args.grad_accumulate_steps} accum "
          f"= {args.train_batch_size * args.grad_accumulate_steps} effective")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 70)

    model = SpeechLLMLightning(
        audio_enc_dim=args.audio_enc_dim,
        llm_dim=args.llm_dim,
        audio_encoder_name=args.audio_encoder_name,
        connector_name=args.connector_name,
        llm_name=args.llm_name,
        finetune_encoder=args.finetune_encoder,
        finetune_llm=args.finetune_llm,
        connector_k=args.connector_k,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_lr=args.max_lr,
        total_training_step=args.total_training_step,
        warmup_steps=args.warmup_steps,
    )

    # Optionally load pretrained weights (supports both adapter-only and full)
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        load_trainable_state_dict(model, args.pretrained_checkpoint)
        print("✅ Pretrained checkpoint loaded.")

    # Print trainable param count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
          f"({100 * trainable_params / total_params:.1f}%)")

    # ---- Build data ----
    collator = MyCollator(args.audio_encoder_name, model.llm_tokenizer)

    train_loader = build_pooled_dataloader(
        args.csv_train_dir, collator,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True, mode="train",
    )
    val_loader = build_pooled_dataloader(
        args.csv_dev_dir, collator,
        batch_size=1,
        num_workers=min(args.num_workers, 2),
        shuffle=False, mode="test",
    )

    # ---- Callbacks ----
    os.makedirs(args.output_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.run_name}" + "-epoch{epoch:02d}-wer{val/wer:.4f}",
            monitor="val/wer",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        AdapterCheckpoint(output_dir=args.output_dir),
        LearningRateMonitor(logging_interval="step"),
        LogProgressBar(refresh_rate=args.log_every_n_steps),
    ]

    # ---- Logger ----
    logger = True  # default PL logger (TensorBoard)
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            reinit=True,
        )

    # ---- Devices ----
    devices = args.devices
    if devices != "auto":
        if "," in str(devices):
            devices = [int(d) for d in str(devices).split(",")]
        else:
            devices = int(devices)

    # ---- Trainer ----
    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.grad_accumulate_steps,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        default_root_dir=args.output_dir,
    )
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches

    trainer = pl.Trainer(**trainer_kwargs)

    # ---- Train ----
    print(f"\n🚀 Starting centralized training\n")
    trainer.fit(model, train_loader, val_loader)

    # ---- Save final adapter (LoRA + connector only) ----
    final_path = os.path.join(args.output_dir, "final_model.pt")
    sd = save_trainable_state_dict(model, final_path)
    print(f"\n✅ Final adapter saved at: {final_path} ({len(sd)} keys)")


if __name__ == "__main__":
    main()
