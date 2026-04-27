#!/usr/bin/env python3
"""
Centralized (non-federated) training of the Voxtral speech-LLM model.

This script trains Voxtral (end-to-end multimodal model) on pooled client
data, mirroring the FL setup for direct comparison.  LoRA adapters are
applied to the language_model Q/K/V/O projections, and the
multi_modal_projector is fully trainable.

Usage:
  # Multi-GPU (4x H100):
  python -m flower_speech_llm.train_centralized_voxtral \
      --config configs/centralized_voxtral.yaml

  # Resume from checkpoint:
  python -m flower_speech_llm.train_centralized_voxtral \
      --config configs/centralized_voxtral.yaml \
      --pretrained-checkpoint centralized_checkpoints_voxtral/adapter-epoch01-step50000-wer0.1500.pt
"""

import argparse
import os
import sys

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

from .model.voxtral import get_voxtral
from .trainer_voxtral import VoxtralLightning
from .dataset_voxtral import VoxtralCSVDataset, VoxtralCollator


class AdapterCheckpoint(pl.Callback):
    """Save adapter-only .pt files after each validation run."""

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
        # Save only trainable parameters (LoRA + connector)
        sd = {k: v.cpu() for k, v in pl_module.state_dict().items()
              if any(p.requires_grad for n, p in pl_module.named_parameters() if n == k)}
        torch.save(sd, path)
        print(f"  💾 Adapter checkpoint saved: {path} ({len(sd)} keys)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Centralized training of Voxtral speech-LLM"
    )

    # ---- Config file ----
    p.add_argument("--config", default="",
                   help="Path to a YAML config file. CLI args override config values.")

    # ---- Model ----
    p.add_argument("--voxtral-model-name", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--model-cache-dir", default="")
    p.add_argument("--data-language", default="en")

    # ---- LoRA ----
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--finetune-encoder", action="store_true", default=False)

    # ---- Optimizer ----
    p.add_argument("--max-lr", type=float, default=1e-4)
    p.add_argument("--total-training-step", type=int, default=2_000_000)
    p.add_argument("--warmup-steps", type=int, default=1000)

    # ---- Training ----
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--grad-accumulate-steps", type=int, default=4)
    p.add_argument("--gradient-clip-val", type=float, default=1.0)
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--val-check-interval", type=float, default=0.25)

    # ---- Data ----
    p.add_argument("--csv-train-dir",
                   default="flower_speech_llm/fl_A1_mixed_316")
    p.add_argument("--csv-dev-dir",
                   default="flower_speech_llm/fl_MLS_dev_speaker")

    # ---- Checkpoint / resume ----
    p.add_argument("--pretrained-checkpoint", default="")
    p.add_argument("--output-dir", default="centralized_checkpoints_voxtral")

    # ---- Hardware ----
    p.add_argument("--devices", default="auto")
    p.add_argument("--strategy", default="auto")
    p.add_argument("--precision", default="bf16-mixed")

    # ---- Logging ----
    p.add_argument("--wandb-project", default="")
    p.add_argument("--run-name", default="centralized-voxtral")

    # ---- Two-pass parse: load YAML defaults, then let CLI override ----
    preliminary, _ = p.parse_known_args()
    if preliminary.config and os.path.exists(preliminary.config):
        with open(preliminary.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        defaults = {k.replace("-", "_"): v for k, v in yaml_cfg.items()}
        for k, v in defaults.items():
            if v is None:
                defaults[k] = None
        p.set_defaults(**defaults)
        print(f"Loaded config from: {preliminary.config}")

    return p.parse_args()


def build_pooled_dataloader(csv_dir, collator, batch_size, num_workers, shuffle):
    """Load all CSV files in csv_dir, merge into ConcatDataset, return DataLoader."""
    csv_files = sorted(
        os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in: {csv_dir}")

    datasets = [VoxtralCSVDataset(csv_file=f) for f in csv_files]
    combined = ConcatDataset(datasets)
    print(f"✅ Pooled {len(csv_files)} CSVs → {len(combined)} samples")

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

    # ---- Build model ----
    print("=" * 70)
    print("Centralized Voxtral Training")
    print("=" * 70)
    print(f"Model:        {args.voxtral_model_name}")
    print(f"LoRA:         r={args.lora_r}, alpha={args.lora_alpha}" if args.use_lora else "LoRA: off")
    print(f"Finetune enc: {args.finetune_encoder}")
    print(f"Train dir:    {args.csv_train_dir}")
    print(f"Dev dir:      {args.csv_dev_dir}")
    print(f"Max epochs:   {args.max_epochs}")
    print(f"Batch size:   {args.train_batch_size} × {args.grad_accumulate_steps} accum "
          f"= {args.train_batch_size * args.grad_accumulate_steps} effective")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 70)

    cache_dir = args.model_cache_dir if args.model_cache_dir else ""
    processor, model = get_voxtral(
        model_name=args.voxtral_model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        finetune_encoder=args.finetune_encoder,
        cache_dir=cache_dir,
    )

    lightning_model = VoxtralLightning(
        model=model,
        processor=processor,
        max_lr=args.max_lr,
        warmup_steps=args.warmup_steps,
        total_training_step=args.total_training_step,
    )

    # Optionally load pretrained weights
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        state_dict = torch.load(args.pretrained_checkpoint, map_location="cpu")
        lightning_model.load_state_dict(state_dict, strict=False)
        print("✅ Pretrained checkpoint loaded.")

    # Print trainable param count
    total_params = sum(p.numel() for p in lightning_model.parameters())
    trainable_params = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable "
          f"({100 * trainable_params / total_params:.1f}%)")

    # ---- Build data ----
    collator = VoxtralCollator(
        processor=processor,
        model_id=args.voxtral_model_name,
        language=args.data_language,
    )

    train_loader = build_pooled_dataloader(
        args.csv_train_dir, collator,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_pooled_dataloader(
        args.csv_dev_dir, collator,
        batch_size=1,
        num_workers=min(args.num_workers, 2),
        shuffle=False,
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
    ]

    # ---- Logger ----
    logger = True
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
    print(f"\n🚀 Starting centralized Voxtral training\n")
    trainer.fit(lightning_model, train_loader, val_loader)

    # ---- Save final adapter (LoRA + connector only) ----
    final_path = os.path.join(args.output_dir, "final_model.pt")
    sd = {k: v.cpu() for k, v in lightning_model.state_dict().items()
          if any(p.requires_grad for n, p in lightning_model.named_parameters() if n == k)}
    torch.save(sd, final_path)
    print(f"\n✅ Final adapter saved at: {final_path} ({len(sd)} keys)")


if __name__ == "__main__":
    main()
