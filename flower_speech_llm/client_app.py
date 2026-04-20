"""speech_llm_fl: A Flower / PyTorch Lightning Federated Learning ClientApp for SpeechLLM."""

import copy
import gc
import random
from collections import OrderedDict

import numpy as np
import torch
import pytorch_lightning as pl
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .trainer import SpeechLLMLightning
from .dataset import MyCollator, build_dataloaders_from_csvs

from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings("ignore")

# ---------> Tensor Core Optimization <---------
torch.set_float32_matmul_precision("medium")

# ---------> Seed Fix <---------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------> Flower ClientApp <---------
app = ClientApp()


# ---------> Helper: Trainable Parameter Utilities <---------

def get_trainable_parameters(model):
    """Extract only trainable parameters as numpy arrays."""
    params, names = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.detach().cpu().numpy())
            names.append(name)
    return params, names


def set_trainable_parameters(model, parameters):
    """Load only trainable parameters back into the model."""
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if len(parameters) != len(trainable_names):
        raise ValueError(
            f"Expected {len(trainable_names)} parameters, got {len(parameters)}"
        )
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.array(v)) if np.array(v).shape != () else torch.tensor([0.0])
            for k, v in zip(trainable_names, parameters)
        }
    )
    model.load_state_dict(state_dict, strict=False)


def state_dict_to_arrays(model):
    """Return trainable parameters as a list of numpy arrays."""
    params, _ = get_trainable_parameters(model)
    return params


def arrays_to_state_dict(model, arrays):
    """Set trainable parameters from a list of numpy arrays."""
    set_trainable_parameters(model, arrays)


# ---------> Build Model from Context <---------

def build_model(context: Context) -> SpeechLLMLightning:
    """Instantiate SpeechLLMLightning from run_config."""
    cfg = context.run_config
    model_config = {
        "audio_enc_dim":        int(cfg.get("audio-enc-dim", 1024)),
        "llm_dim":              int(cfg.get("llm-dim", 2048)),
        "audio_encoder_name":   cfg.get("audio-encoder-name", "microsoft/wavlm-large"),
        "connector_name":       cfg.get("connector-name", "linear"),
        "llm_name":             cfg.get("llm-name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "finetune_encoder":     bool(cfg.get("finetune-encoder", False)),
        "connector_k":          int(cfg.get("connector-k", 2)),
        "use_lora":             bool(cfg.get("use-lora", True)),
        "lora_r":               int(cfg.get("lora-r", 8)),
        "lora_alpha":           int(cfg.get("lora-alpha", 16)),
        "max_lr":               float(cfg.get("max-lr", 1e-4)),
        "total_training_step":  int(cfg.get("total-training-step", 10_000_000)),
        "warmup_steps":         int(cfg.get("warmup-steps", 100)),
    }
    return SpeechLLMLightning(**model_config)


# ---------> Build DataLoaders from Context <---------

def build_loaders(model, context: Context):
    """Build train and val DataLoaders for this client partition."""
    cfg = context.run_config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    tokenizer = model.llm_tokenizer
    audio_encoder_name = cfg.get("audio-encoder-name", "microsoft/wavlm-large")
    collator = MyCollator(audio_encoder_name, tokenizer)

    csv_train_dir = cfg.get("csv-train-dir", "./fl_multilingual")
    csv_dev_dir   = cfg.get("csv-dev-dir",   "./fl_MLS_dev_speaker")

    train_loaders = build_dataloaders_from_csvs(
        csv_dir=csv_train_dir, my_collator=collator,
        batch_size=int(cfg.get("train-batch-size", 4)),
        num_workers=int(cfg.get("num-workers", 3)),
        shuffle=True,
    )
    dev_loaders = build_dataloaders_from_csvs(
        csv_dir=csv_dev_dir, my_collator=collator,
        batch_size=1, num_workers=3, shuffle=False,
    )

    train_loader = train_loaders[partition_id % len(train_loaders)]
    val_loader   = dev_loaders[partition_id % len(dev_loaders)]
    return train_loader, val_loader


# ---------> FedProx Callback <---------

class FedProxCallback(pl.Callback):
    """Adds a proximal term (mu/2 * ||w - w_global||^2) to the training loss."""

    def __init__(self, global_params: list, mu: float):
        super().__init__()
        self.global_params = global_params  # list of tensors (global model weights)
        self.mu = mu

    def on_before_backward(self, trainer, pl_module, loss):
        if self.mu <= 0:
            return
        proximal_term = 0.0
        for local_p, global_p in zip(
            (p for p in pl_module.parameters() if p.requires_grad),
            self.global_params,
        ):
            proximal_term += ((local_p - global_p.to(local_p.device)) ** 2).sum()
        loss += (self.mu / 2.0) * proximal_term


# ---------> Train Handler <---------

@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train SpeechLLM locally and return updated weights + metrics."""

    # Build model and load weights received from server
    model = build_model(context)
    arrays = msg.content["arrays"].to_numpy_ndarrays()   # list[np.ndarray]
    set_trainable_parameters(model, arrays)

    # Load data for this partition
    train_loader, _ = build_loaders(model, context)

    # Read training config sent by the server (with fallbacks to run_config)
    cfg = context.run_config
    server_cfg = msg.content.get("config", {})
    if "local-epochs" not in server_cfg:
        print("⚠️  'local-epochs' not found in server config — using fallback from run_config")
    if "lr" not in server_cfg:
        print("⚠️  'lr' not found in server config — using fallback from run_config")
    local_epochs      = int(server_cfg.get("local-epochs", cfg.get("local-epochs", 10)))
    lr                = float(server_cfg.get("lr", cfg.get("max-lr", 1e-4)))
    train_batch_limit = int(cfg.get("train-batch-per-epoch", 200))
    grad_accum        = int(cfg.get("grad-accumulate-steps", 4))
    fedprox_mu        = float(cfg.get("fedprox-mu", 0.0))

    # Apply learning rate from server
    model.max_lr = lr

    # FedProx: save a copy of global weights for proximal term
    callbacks = []
    if fedprox_mu > 0:
        global_params = [
            p.detach().clone() for p in model.parameters() if p.requires_grad
        ]
        callbacks.append(FedProxCallback(global_params, fedprox_mu))
        print(f"  [FedProx] mu={fedprox_mu}")

    # WandB logger — all clients log under the same project, grouped by partition
    partition_id = context.node_config.get("partition-id", 0)
    wandb_logger = WandbLogger(
        project="speech-llm-fl",
        name=f"client-{partition_id}",
        group=cfg.get("csv-train-dir", "default"),
        reinit=True,
        settings={"quiet": True},
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=local_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=wandb_logger,
        enable_progress_bar=True,
        limit_train_batches=train_batch_limit,
        accumulate_grad_batches=grad_accum,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader)

    # Collect metrics
    fit_metrics = trainer.callback_metrics
    train_loss = float(fit_metrics.get("train/loss", 0.0))
    num_examples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 1000

    # Clean up
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    # Build reply Message
    updated_arrays = state_dict_to_arrays(model)
    array_record  = ArrayRecord(updated_arrays)
    metric_record = MetricRecord({"train_loss": train_loss, "num-examples": float(num_examples), "lr": lr})
    content = RecordDict({"arrays": array_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# ---------> Evaluate Handler <---------

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate SpeechLLM on local validation data and return metrics."""

    # Build model and load weights
    model = build_model(context)
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    set_trainable_parameters(model, arrays)

    # Load data
    _, val_loader = build_loaders(model, context)

    # Evaluate
    partition_id = context.node_config.get("partition-id", 0)
    wandb_logger = WandbLogger(
        project="speech-llm-fl",
        name=f"eval-client-{partition_id}",
        group=cfg.get("csv-train-dir", "default"),
        reinit=True,
        settings={"quiet": True},
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=wandb_logger,
        enable_progress_bar=True,
        limit_val_batches=1,
    )
    results = trainer.validate(model, val_loader)

    loss = float(results[0].get("val/loss", 0.0))
    wer  = float(results[0].get("val/wer",  0.0))
    num_examples = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 100

    # Build reply Message
    metric_record = MetricRecord({"eval_loss": loss, "wer": wer, "num-examples": float(num_examples)})
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
