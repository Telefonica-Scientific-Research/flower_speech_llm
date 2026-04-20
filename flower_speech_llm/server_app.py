"""speech_llm_fl: A Flower / PyTorch Lightning Federated Learning ServerApp for SpeechLLM."""

import gc
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from flwr.app import ArrayRecord, Context
from flwr.common.record import ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .trainer import SpeechLLMLightning

import warnings
warnings.filterwarnings("ignore")

# ---------> Flower ServerApp <---------
app = ServerApp()


# ---------> Helper: Trainable Parameter Utilities <---------

def get_trainable_parameters(model):
    """Extract only trainable parameters as numpy arrays."""
    params, names = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param.detach().cpu().numpy())
            names.append(name)
    return params, names


def load_model_from_config(cfg: dict) -> SpeechLLMLightning:
    """Instantiate SpeechLLMLightning from a config dict."""
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


# ---------> Custom FedAvg Strategy (Learning Rate Decay + Checkpointing) <---------

class SpeechLLMFedAvg(FedAvg):
    """
    Custom FedAvg strategy that adds:
      - Per-round learning rate decay sent to clients via config
      - Model checkpoint saving after each aggregation round
      - Optional pretrained checkpoint loading at initialization
    """

    def __init__(self, model_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = model_cfg
        self.initial_lr      = float(model_cfg.get("max-lr", 1e-4))
        self.decay_factor    = float(model_cfg.get("lr-decay-factor", 0.9))
        self.decay_every     = int(model_cfg.get("lr-decay-every", 10))
        self.local_epochs    = int(model_cfg.get("local-epochs", 10))
        self.checkpoint_dir  = model_cfg.get("checkpoint-dir", "FL_SLAM_checkpoints")
        self.checkpoint_offset = int(model_cfg.get("checkpoint-offset", 0))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def configure_train(self, server_round, arrays, config, grid):
        """Inject per-round LR decay and local-epochs into the config sent to clients."""
        current_lr = self.initial_lr * (self.decay_factor ** (server_round // self.decay_every))
        print(f"[Round {server_round}] Learning rate: {current_lr:.6f}")
        config["lr"] = current_lr
        config["local-epochs"] = self.local_epochs
        return super().configure_train(server_round, arrays, config, grid)

    def on_aggregate(self, server_round: int, aggregated_arrays: list) -> None:
        """Save a checkpoint after each aggregation round."""
        model = load_model_from_config(self.model_cfg)
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

        if len(aggregated_arrays) != len(trainable_names):
            print(
                f"[Round {server_round}] ⚠️  Mismatch: "
                f"{len(aggregated_arrays)} aggregated tensors vs {len(trainable_names)} trainable params."
            )
            return

        state_dict = OrderedDict(
            {k: torch.tensor(np.array(v)) for k, v in zip(trainable_names, aggregated_arrays)}
        )
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f"[Round {server_round}] ❌ Error loading state_dict: {e}")
            return

        round_id = server_round + self.checkpoint_offset
        ckpt_path = os.path.join(self.checkpoint_dir, f"Checkpoint-round-{round_id}.ckpt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Round {server_round}] ✅ Aggregated model saved at {ckpt_path}")

        del model, state_dict
        gc.collect()


# ---------> ServerApp Main <---------

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    cfg = context.run_config
    num_rounds   = int(cfg.get("num-server-rounds", 200))
    pretrained   = cfg.get("pretrained-checkpoint", "")
    checkpoint_dir   = cfg.get("checkpoint-dir", "FL_SLAM_checkpoints")
    checkpoint_offset = int(cfg.get("checkpoint-offset", 0))
    fraction_fit     = float(cfg.get("fraction-fit", 0.3))
    fraction_eval    = float(cfg.get("fraction-evaluate", 0.0))
    min_fit_clients  = int(cfg.get("min-fit-clients", 2))
    min_eval_clients = int(cfg.get("min-evaluate-clients", 2))

    # ---- Load global model ----
    global_model = load_model_from_config(cfg)

    # Optionally load from pretrained checkpoint
    if pretrained and os.path.exists(pretrained):
        print(f"Loading pretrained checkpoint from: {pretrained}")
        state_dict = torch.load(pretrained, map_location="cpu")
        global_model.load_state_dict(state_dict)
        torch.cuda.empty_cache()
        print("✅ Pretrained checkpoint loaded.")
    else:
        print("No pretrained checkpoint found — starting from random init.")

    # Extract only trainable parameters for federation
    trainable_params, _ = get_trainable_parameters(global_model)
    initial_arrays = ArrayRecord(trainable_params)

    # ---- Build strategy ----
    strategy = SpeechLLMFedAvg(
        model_cfg=dict(cfg),
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_train_nodes=min_fit_clients,
        min_evaluate_nodes=min_eval_clients,
    )

    # ---- Run federation ----
    print(f"\n🚀 Starting Federated Learning — {num_rounds} rounds, {int(1/fraction_fit):.0f}x sampling ratio\n")
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # ---- Save final model ----
    final_arrays = result.arrays.to_numpy_ndarrays()
    trainable_names = [n for n, p in global_model.named_parameters() if p.requires_grad]
    state_dict = OrderedDict(
        {k: torch.tensor(np.array(v)) for k, v in zip(trainable_names, final_arrays)}
    )
    global_model.load_state_dict(state_dict, strict=False)

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(global_model.state_dict(), final_path)
    print(f"\n✅ Final model saved at: {final_path}")
