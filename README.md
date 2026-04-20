# Federated SpeechLLM with Flower

Federated learning for Speech LLMs with Flower and PyTorch Lightning. Trains a WavLM + TinyLlama model with LoRA across distributed clients on the Multilingual LibriSpeech corpus. Supports FedAvg/FedProx strategies, speaker-based and IID partitioning, per-round LR decay, W&B logging, and multi-GPU Ray simulation. Only adapter weights are shared — raw audio never leaves the client.

> **Based on** [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih** — the original Flower Hub app for federated SpeechLLM training. This repository extends the original with compatibility fixes for Flower 1.29+, new experiment tooling, and additional features. See [Changes from Original](#changes-from-original) for full details.

---

## Key Features

- 🎙️ **Multimodal Architecture** — WavLM audio encoder + connector + TinyLlama LLM for end-to-end speech understanding (transcription, gender, emotion, age, accent, speech activity)
- 🔒 **Privacy-Preserving** — Only LoRA weights + connector are shared; raw audio never leaves the client
- 🔁 **FedAvg + FedProx** — Custom `SpeechLLMFedAvg` strategy with per-round LR decay, configurable client sampling, and automatic checkpointing. Optional FedProx proximal term for non-IID robustness
- ⚡ **LoRA Fine-Tuning** — Parameter-efficient federation via PEFT, drastically reducing communication cost
- 📊 **W&B Logging** — Per-client training metrics and validation predictions logged to Weights & Biases
- 💾 **Checkpoint Resumption** — Resume from any `.ckpt` via config
- 🖥️ **Multi-GPU Simulation** — Ray backend with configurable client-to-GPU mapping and Tensor Core optimization
- 🌍 **Multilingual Data Pipeline** — Scripts to download MLS (8 languages), create IID or speaker-based FL partitions

---

## Changes from Original

This repository is a fork/extension of [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih**. The following changes were made to support **Flower 1.29+**, add missing components, and extend functionality.

### API Compatibility (Flower 1.29+)

The original app targeted an older Flower version. These changes were required to run on `flwr>=1.29`:

| Change | Files |
|--------|-------|
| `ArrayRecord.to_numpy_arrays()` → `.to_numpy_ndarrays()` | `client_app.py`, `server_app.py` |
| `FedAvg(min_fit_clients=...)` → `FedAvg(min_train_nodes=...)` | `server_app.py` |
| `FedAvg(min_evaluate_clients=...)` → `FedAvg(min_evaluate_nodes=...)` | `server_app.py` |
| `get_round_config()` → `configure_train()` override | `server_app.py` |
| Added `from flwr.common.record import ConfigRecord` | `server_app.py` |
| Bare imports → relative imports (`.trainer`, `.dataset`, `.model.*`) | All Python files |
| Project name `speech_llm_fl` → `speech-llm-fl` (hyphens required) | `pyproject.toml` |

### Missing `model/` Package (New)

The original Flower Hub app did **not** include the `model/` subpackage that `trainer.py` imports from. This repo provides it:

| File | Contents |
|------|----------|
| `model/__init__.py` | Package init |
| `model/encoder.py` | `TransformerAudioEncoder` (WavLM/HuBERT/Wav2Vec2 wrapper), `WhisperEncoder`, `get_audio_encoder()` factory |
| `model/connector.py` | `LinearConnector`, `LinearPoolConnector`, `CNNConnector`, `get_connector()` factory |
| `model/llm.py` | `get_llm()` — loads a causal LLM with optional LoRA via PEFT |

### New Features

| Feature | Details |
|---------|---------|
| **FedProx support** | `FedProxCallback` in `client_app.py` — adds proximal term `(mu/2) * \|\|w - w_global\|\|²` to loss. Enable via `fedprox-mu` config |
| **W&B integration** | `WandbLogger` wired into both train and evaluate Trainers, grouped by experiment |
| **Tensor Core optimization** | `torch.set_float32_matmul_precision("medium")` for faster training on Ampere+ GPUs |
| **Safe config reading** | Client reads server config via `.get()` with fallbacks and warning messages instead of hard indexing |

### New Scripts

| Script | Purpose |
|--------|---------|
| `prepare_mls_fl.py` | Download MLS (8 languages) from HuggingFace, save audio as FLAC, create per-speaker CSVs |
| `create_experiment_partitions.py` | Generate IID (mixed-language) and non-IID (one-speaker) FL client partitions using `flwr-datasets` |
| `evaluate_fl_model.py` | Evaluate FL checkpoints on MLS test sets, report per-language WER |
| `run_experiments.sh` | Run A1/B1/B2 experiments sequentially with evaluation |

### Bug Fixes

| Fix | Details |
|-----|---------|
| `TransformerAudioEnoder` → `TransformerAudioEncoder` | Typo in class name (trainer.py, encoder.py) |
| `msg.content["config"]["local-epochs"]` → safe `.get()` chain | Prevents `KeyError` when server config is empty |

---

## Architecture

| File | Description |
|---|---|
| `client_app.py` | `ClientApp` with `@app.train()` and `@app.evaluate()` handlers — loads weights, runs local training, returns updated parameters and metrics |
| `server_app.py` | `ServerApp` with `SpeechLLMFedAvg` strategy — per-round LR decay, aggregation, and checkpoint saving |
| `trainer.py` | `SpeechLLMLightning` — PyTorch Lightning module: full SpeechLLM forward pass, training/validation/test steps, metric logging |
| `dataset.py` | `InstructionalAudioDataset`, `MyCollator`, `build_dataloaders_from_csvs` — CSV-based audio dataset per client |
| `model/` | Audio encoder, connector, and LLM wrappers (see above) |
| `pyproject.toml` | All federation, model, training, data, and checkpoint config |

### Model Architecture

```
Audio Input (waveform)
        │
        ▼
 WavLM Encoder (frozen or finetuned)
        │
        ▼
   Connector (Linear / LinearPool / CNN)
        │
        ▼
  [Pre-prompt embeddings] + [Speech embeddings] + [Post-prompt embeddings]
        │
        ▼
   TinyLlama LLM (LoRA fine-tuned)
        │
        ▼
  Structured JSON output:
  { "Transcript": "...", "Gender": "male", "Emotion": "neutral", ... }
```

### Federated Learning Process

1. **Initialization** — Server loads global `SpeechLLMLightning` model (optionally from a pretrained checkpoint) and extracts only trainable parameters (LoRA + connector)
2. **Round Config** — Server computes a decayed learning rate via `configure_train()` and injects it into the config sent to sampled clients
3. **Local Training** — Each client loads received weights, trains for `local-epochs` × `train-batch-per-epoch` steps with PyTorch Lightning (optionally with FedProx proximal term)
4. **Aggregation** — Server performs weighted FedAvg over client updates proportional to dataset sizes
5. **Checkpointing** — Aggregated model saved to disk after every round; final model saved as `final_model.pt`

---

## Installation

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (strongly recommended)
- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Setup with uv

```bash
# Clone the repository
git clone https://github.com/Telefonica-Scientific-Research/flower_speech_llm.git
cd flower_speech_llm

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate

# Install the project in editable mode
uv pip install -e .
```

Or install all dependencies directly:

```bash
uv pip install \
  "flwr[simulation]>=1.29.0" \
  "pytorch-lightning>=2.0.0" \
  "torch>=2.4.0" \
  "torchaudio>=2.4.0" \
  "transformers>=4.40.0" \
  "datasets>=2.0.0" \
  "pandas>=1.5.0" \
  "jiwer>=3.0.0" \
  "wandb>=0.15.0" \
  "peft>=0.10.0" \
  "flwr-datasets[audio]>=0.6.0" \
  "soundfile>=0.12.0"
```

### Verify installation

```bash
python -c "import flwr; print('Flower:', flwr.__version__)"
python -c "from flower_speech_llm import client_app; print('OK')"
```

---

## Data Preparation

### Download MLS Dataset

```bash
python flower_speech_llm/prepare_mls_fl.py --languages all --output-dir ./mls_audio
```

This downloads Multilingual LibriSpeech (8 languages: English, German, Dutch, French, Spanish, Italian, Portuguese, Polish) and creates per-client(speaker) CSV files.

### Create FL Partitions

```bash
python flower_speech_llm/create_experiment_partitions.py --base-dir ./flower_speech_llm
```

This generates three experiment settings:

| Setting | Description | Clients |
|---------|-------------|---------|
| **A1** (IID) | Multilingual-mixed random partitioning | 316 |
| **B1** (Non-IID) | One speaker per client | 316 |
| **B2** (Non-IID + FedProx) | Same as B1, with FedProx enabled | 316 |

### CSV Format

Each client CSV has the following columns:

| Column | Description |
|---|---|
| `audio_path` | Path to audio file (16kHz, FLAC/WAV) |
| `transcript` | Ground-truth transcription |
| `gender` | Speaker gender |
| `emotion` | Emotion label |
| `age` | Age group |
| `accent` | Accent/language label |
| `isspeech` | Whether audio contains speech |

---

## Run Experiments

### Quick Start

```bash
flwr run . --run-config 'csv-train-dir="./flower_speech_llm/fl_A1_mixed_316" csv-dev-dir="./flower_speech_llm/fl_dev_316"'
```

### Experiment Settings

```bash
# A1: Mixed-multilingual + FedAvg
flwr run . --run-config 'csv-train-dir="./flower_speech_llm/fl_A1_mixed_316" csv-dev-dir="./flower_speech_llm/fl_dev_316"'

# B1: One-speaker + FedAvg
flwr run . --run-config 'csv-train-dir="./flower_speech_llm/fl_B1_speaker_316" csv-dev-dir="./flower_speech_llm/fl_dev_316"'

# B2: One-speaker + FedProx (mu=0.01)
flwr run . --run-config 'csv-train-dir="./flower_speech_llm/fl_B1_speaker_316" csv-dev-dir="./flower_speech_llm/fl_dev_316" fedprox-mu=0.01'
```

### Override Settings at Runtime

```bash
flwr run . --run-config 'num-server-rounds=10 local-epochs=5 max-lr=0.00005'
```

### Resume from Checkpoint

```bash
flwr run . --run-config 'pretrained-checkpoint="FL_SLAM_checkpoints/Checkpoint-round-50.ckpt" checkpoint-offset=50'
```

### Restrict to Specific GPUs

```bash
CUDA_VISIBLE_DEVICES=0 flwr run . --run-config 'csv-train-dir="./flower_speech_llm/fl_A1_mixed_316" csv-dev-dir="./flower_speech_llm/fl_dev_316"'
```

### Monitor a Run

```bash
flwr ls                           # List all runs
flwr log <run_id>                 # Stream logs
flwr stop <run_id>                # Stop a run
```

### Evaluate a Checkpoint

```bash
python flower_speech_llm/evaluate_fl_model.py \
  --checkpoint FL_SLAM_checkpoints/final_model.pt \
  --test-dir flower_speech_llm/fl_MLS_test
```

---

## Configuration

All settings are in `pyproject.toml`. No code changes needed.

### Federation

```toml
num-server-rounds    = 40      # Total FL rounds
fraction-fit         = 0.3     # Fraction of clients sampled per round
fraction-evaluate    = 0.0
min-fit-clients      = 2
```

### Model

```toml
audio-encoder-name = "microsoft/wavlm-large"
llm-name           = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
connector-name     = "linear"       # "linear", "linear-pool", or "cnn"
use-lora           = true
lora-r             = 8
lora-alpha         = 16
```

### Training

```toml
local-epochs          = 10
train-batch-size      = 4
train-batch-per-epoch = 200
grad-accumulate-steps = 4
max-lr                = 0.0001
```

### FedProx

```toml
fedprox-mu = 0.0    # Set > 0 to enable (e.g. 0.01)
```

### LR Decay

```toml
lr-decay-factor = 0.9    # LR at round r = max-lr × factor^(r // decay-every)
lr-decay-every  = 10
```

---

## Project Structure

```
flower_speech_llm/
├── flower_speech_llm/
│   ├── __init__.py
│   ├── client_app.py                   # Flower ClientApp (train + evaluate)
│   ├── server_app.py                   # Flower ServerApp (SpeechLLMFedAvg)
│   ├── trainer.py                      # SpeechLLMLightning model
│   ├── dataset.py                      # Dataset and dataloader utilities
│   ├── model/
│   │   ├── encoder.py                  # WavLM / Whisper encoder wrappers
│   │   ├── connector.py                # Linear / LinearPool / CNN connectors
│   │   └── llm.py                      # LLM + LoRA loader
│   ├── prepare_mls_fl.py              # MLS dataset download + preparation
│   ├── create_experiment_partitions.py # FL partition generator
│   ├── evaluate_fl_model.py           # Checkpoint evaluation script
│   └── run_experiments.sh             # Batch experiment runner
├── pyproject.toml                      # Project config + Flower settings
└── README.md
```

---

## Metrics Tracked

| Metric | Description |
|---|---|
| `train_loss` | Cross-entropy loss on local training data |
| `val/loss` | Validation loss |
| `val/wer` | Word Error Rate on transcript predictions |
| `val/gender` | Gender classification accuracy |
| `val/emotion` | Emotion classification accuracy |
| `val/age` | Age group classification accuracy |
| `val/accent` | Accent classification accuracy |
| `val/speech_activity` | Speech activity detection accuracy |

---

## Results

Performance comparison of **WavLM** vs. **Whisper** encoders, measured in Word Error Rate (WER ↓) on LibriSpeech (LS) and Multilingual LibriSpeech (MLS) test sets. Central training serves as the upper bound.

| Setting | WavLM (Round=100) LS | WavLM (Round=100) MLS | Whisper (Round=40) LS | Whisper (Round=40) MLS |
|---|---|---|---|---|
| **Central Training** ⭐ | **6.1** | **18.4** | 6.4 | **16.4** |
| FL Sample Cluster | 9.7 | 19.6 | 7.7 | 16.4 |

> ⭐ Central training is the upper bound (non-federated). Lower WER is better.

---

## Credits

- **Original app**: [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih** — Federated SpeechLLM with Flower
- **Flower Framework**: [flower.ai](https://flower.ai/)
- **WavLM**: [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- **TinyLlama**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **LoRA / PEFT**: [huggingface/peft](https://github.com/huggingface/peft)
- **PyTorch Lightning**: [lightning.ai](https://lightning.ai/docs/pytorch/stable/)

---

## License

Apache License 2.0