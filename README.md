# Flower Speech-LLM: Federated Learning for Speech Language Models

Federated learning for Speech LLMs using [Flower](https://flower.ai/) and PyTorch Lightning. Supports two model architectures — **WavLM + TinyLlama** (modular encoder-connector-LLM pipeline) and **Voxtral** (end-to-end multimodal model) — trained with LoRA across distributed clients on the Multilingual LibriSpeech corpus. Only adapter weights are shared; raw audio never leaves the client.

> **Based on** [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih** — the original Flower Hub app for federated SpeechLLM training. This repository extends the original with Flower 1.29+ compatibility, Voxtral support, HPC deployment, and additional features. See [Changes from Original](#changes-from-original) for details.

---

## Supported Models

### Speech-LLM (Encoder + Connector + LLM)

Modular three-stage pipeline for speech understanding tasks (transcription, gender, emotion, age, accent, speech activity). The audio encoder and LLM are configurable — any HuggingFace-compatible encoder (WavLM, HuBERT, Wav2Vec2, Whisper) and causal LLM can be used.

```
Audio (waveform) → Audio Encoder (frozen) → Connector (Linear) → LLM (LoRA)
                                                                     ↓
                                           { "Transcript": "...", "Gender": "male", ... }
```

- **Encoder**: Any HuggingFace audio encoder — default: `microsoft/wavlm-large` (1024-dim, frozen by default). Also supports HuBERT, Wav2Vec2, and Whisper via `audio-encoder-name` config
- **Connector**: Linear / LinearPool / CNN (fully trainable)
- **LLM**: Any HuggingFace causal LLM — default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with LoRA (r=8, alpha=16). Configurable via `llm-name`

### Voxtral (End-to-End Multimodal)

[Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) — Mistral's end-to-end speech-language model. No separate encoder or connector needed.

```
Audio (waveform) → Voxtral Audio-Encoder (frozen) → Multi-Modal Projector (trainable) → LLM (LoRA)
                                                                                            ↓
                                                                                      Transcription
```

- **Audio-Encoder**: Frozen (optional LoRA with `finetune-encoder = true`)
- **Projector**: Fully trainable multi-modal connector
- **LLM**: Mistral-based language model with LoRA on Q/K/V/O projections
- **Multilingual**: Per-sample language tags used in prompts

---

## Key Features

- **Two model backends** — Switch between `speech-llm` and `voxtral` via a single config key
- **Privacy-preserving** — Only LoRA + connector weights are shared; raw audio never leaves the client
- **FedAvg + FedProx** — Custom `SpeechLLMFedAvg` strategy with per-round LR decay, configurable client sampling, and automatic checkpointing
- **LoRA fine-tuning** — Parameter-efficient federation via PEFT, reducing communication cost
- **W&B logging** — Per-client training metrics and validation predictions logged to Weights & Biases
- **Checkpoint resumption** — Resume from any `.ckpt` or `.pt` via config
- **Multi-GPU simulation** — Ray backend with configurable client-to-GPU mapping and Tensor Core optimization
- **Multilingual data pipeline** — Scripts to download MLS (8 languages), create IID or speaker-based FL partitions
- **HPC deployment** — Singularity/Docker images with pre-downloaded models for offline compute nodes

---

## Changes from Original

This repository extends [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih**:

### API Compatibility (Flower 1.29+)

| Change | Files |
|--------|-------|
| `ArrayRecord.to_numpy_arrays()` → `.to_numpy_ndarrays()` | `client_app.py`, `server_app.py` |
| `FedAvg(min_fit_clients=...)` → `FedAvg(min_train_nodes=...)` | `server_app.py` |
| `get_round_config()` → `configure_train()` override | `server_app.py` |
| Bare imports → relative imports | All Python files |

### New: `model/` Package

The original app did **not** include the `model/` subpackage. This repo provides:

| File | Contents |
|------|----------|
| `model/encoder.py` | `TransformerAudioEncoder` (WavLM/HuBERT/Wav2Vec2), `WhisperEncoder`, `get_audio_encoder()` |
| `model/connector.py` | `LinearConnector`, `LinearPoolConnector`, `CNNConnector`, `get_connector()` |
| `model/llm.py` | `get_llm()` — causal LLM with optional LoRA via PEFT |
| `model/voxtral.py` | `get_voxtral()` — loads Voxtral with frozen audio tower, trainable projector, LoRA on LLM |

### New: Voxtral Support

| File | Purpose |
|------|---------|
| `model/voxtral.py` | Model loader: freezes audio tower, unfreezes projector, applies LoRA to LLM |
| `trainer_voxtral.py` | `VoxtralLightning` — PyTorch Lightning wrapper for Voxtral training/validation |
| `dataset_voxtral.py` | `VoxtralCSVDataset`, `VoxtralCollator` — per-sample language-tagged data pipeline |
| `evaluate_fl_model.py` | Updated with `--model-type voxtral` for Voxtral evaluation |
| `client_app.py` | Model routing via `model-type` config key |
| `server_app.py` | Server-side model routing and weight initialization |

### New Features

| Feature | Details |
|---------|---------|
| **FedProx** | Proximal term callback — enable via `fedprox-mu > 0` |
| **W&B integration** | Per-client logging grouped by experiment |
| **Tensor Core optimization** | `float32_matmul_precision("medium")` for Ampere+ GPUs |
| **HPC containers** | Singularity `.def` + Dockerfile + SLURM script in `deploy/` |

### New Scripts

| Script | Purpose |
|--------|---------|
| `prepare_mls_fl.py` | Download MLS (8 languages), save audio as FLAC, create per-speaker CSVs |
| `create_experiment_partitions.py` | Generate IID and speaker-based FL partitions using `flwr-datasets` |
| `evaluate_fl_model.py` | Evaluate checkpoints on MLS test sets (supports both model types) |
| `run_experiments.sh` | Run A1/B1/B2 experiments sequentially |

---

## Architecture

| File | Description |
|---|---|
| `client_app.py` | Flower `ClientApp` — routes to speech-llm or voxtral, local training, returns updated parameters |
| `server_app.py` | Flower `ServerApp` — `SpeechLLMFedAvg` strategy with LR decay and checkpointing |
| `trainer.py` | `SpeechLLMLightning` — WavLM + connector + TinyLlama pipeline |
| `trainer_voxtral.py` | `VoxtralLightning` — end-to-end Voxtral wrapper |
| `dataset.py` | CSV-based audio dataset for speech-llm (multi-task JSON output) |
| `dataset_voxtral.py` | CSV-based audio dataset for Voxtral (language-tagged transcription) |
| `model/` | Encoder, connector, LLM, and Voxtral model loaders |
| `evaluate_fl_model.py` | Checkpoint evaluation (both model types) |

### Federated Learning Process

1. **Initialization** — Server loads global model (speech-llm or voxtral) and extracts trainable parameters (LoRA + connector/projector)
2. **Round Config** — Server computes decayed LR via `configure_train()` and sends to sampled clients
3. **Local Training** — Each client trains for `local-epochs` × `train-batch-per-epoch` steps (optionally with FedProx)
4. **Aggregation** — Weighted FedAvg over client updates proportional to dataset sizes
5. **Checkpointing** — Model saved after every round; final model saved as `final_model.pt`

---

## Installation

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (strongly recommended)

### Setup

```bash
git clone https://github.com/Telefonica-Scientific-Research/flower_speech_llm.git
cd flower_speech_llm

# Option A: uv (recommended)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Verify

```bash
python -c "from flower_speech_llm.client_app import app; print('OK')"
```

---

## Data Preparation

### Download MLS Dataset

```bash
python flower_speech_llm/prepare_mls_fl.py --languages all --output-dir ./mls_audio
```

Downloads Multilingual LibriSpeech (8 languages: English, German, Dutch, French, Spanish, Italian, Portuguese, Polish) and creates per-speaker CSV files.

### Create FL Partitions

```bash
python flower_speech_llm/create_experiment_partitions.py --base-dir ./flower_speech_llm
```

| Setting | Description | Clients |
|---------|-------------|---------|
| **A1** (IID) | Multilingual-mixed random partitioning | 316 |
| **B1** (Non-IID) | One speaker per client | 316 |
| **B2** (Non-IID + FedProx) | Same as B1, with FedProx enabled | 316 |

### CSV Format

**Speech-LLM** requires: `audio_path`, `transcript`, `gender`, `emotion`, `age`, `accent`, `isspeech`

**Voxtral** requires: `audio_path`, `transcript`, `language`

---

## Run Experiments

### Speech-LLM (default)

```bash
# A1: Mixed-multilingual IID + FedAvg
flwr run . --run-config 'csv-train-dir="/path/to/fl_A1_mixed_316" csv-dev-dir="/path/to/fl_dev_316"'

# B1: One-speaker non-IID + FedAvg
flwr run . --run-config 'csv-train-dir="/path/to/fl_B1_speaker_316" csv-dev-dir="/path/to/fl_dev_316"'

# B2: One-speaker non-IID + FedProx
flwr run . --run-config 'csv-train-dir="/path/to/fl_B1_speaker_316" csv-dev-dir="/path/to/fl_dev_316" fedprox-mu=0.01'
```

### Voxtral

```bash
flwr run . --run-config 'model-type="voxtral" csv-train-dir="/path/to/fl_A1_mixed_316" csv-dev-dir="/path/to/fl_dev_316"'
```

### Override Settings

```bash
flwr run . --run-config 'num-server-rounds=10 local-epochs=5 lora-r=16 max-lr=0.00005'
```

### Resume from Checkpoint

```bash
flwr run . --run-config 'pretrained-checkpoint="/path/to/Checkpoint-round-50.ckpt" checkpoint-offset=50'
```

### Monitor

```bash
flwr ls                           # List all runs
flwr log <run_id>                 # Stream logs
flwr stop <run_id>                # Stop a run
```

### Evaluate

```bash
# Speech-LLM
python flower_speech_llm/evaluate_fl_model.py \
  --checkpoint FL_SLAM_checkpoints/final_model.pt \
  --test-dir flower_speech_llm/fl_MLS_test

# Voxtral
python flower_speech_llm/evaluate_fl_model.py \
  --model-type voxtral \
  --checkpoint FL_SLAM_checkpoints/final_model.pt \
  --test-dir flower_speech_llm/fl_MLS_test
```

---

## Configuration

All settings in `pyproject.toml` under `[tool.flwr.app.config]`:

### Model Selection

```toml
model-type = "speech-llm"     # "speech-llm" or "voxtral"
```

### Speech-LLM Config

```toml
audio-encoder-name = "microsoft/wavlm-large"
llm-name           = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
connector-name     = "linear"       # "linear", "linear-pool", or "cnn"
use-lora           = true
lora-r             = 8
lora-alpha         = 16
finetune-encoder   = false
```

### Voxtral Config

```toml
voxtral-model-name = "mistralai/Voxtral-Mini-3B-2507"
data-language      = "en"           # Default language for prompts
model-cache-dir    = ""             # HuggingFace cache directory
# lora-r, lora-alpha, use-lora, finetune-encoder are shared
```

### Federation

```toml
num-server-rounds  = 40
fraction-fit       = 0.3
fraction-evaluate  = 0.0
min-fit-clients    = 2
```

### Training

```toml
local-epochs          = 10
train-batch-size      = 4
train-batch-per-epoch = 200
grad-accumulate-steps = 4
max-lr                = 0.0001
```

### FedProx & LR Decay

```toml
fedprox-mu      = 0.0       # Set > 0 to enable (e.g. 0.01)
lr-decay-factor = 0.9       # LR at round r = max-lr × factor^(r // decay-every)
lr-decay-every  = 10
```

---

## HPC Deployment

Pre-built container images for offline HPC nodes (no internet required at runtime). All models are baked in.

### Build

```bash
# Option A: Singularity (direct)
singularity build flower_speech_llm.sif deploy/flower_speech_llm.def

# Option B: Docker → Singularity
docker build -t flower_speech_llm:latest -f deploy/Dockerfile .
docker save flower_speech_llm:latest -o flower_speech_llm.tar
singularity build flower_speech_llm.sif docker-archive://flower_speech_llm.tar
```

### Run on SLURM

```bash
# Edit paths in deploy/run_bsc.sh, then:
sbatch deploy/run_bsc.sh
```

See [`deploy/`](deploy/) for Singularity definition, Dockerfile, and SLURM script.

---

## Project Structure

```
flower_speech_llm/
├── flower_speech_llm/
│   ├── client_app.py                   # Flower ClientApp (train + evaluate)
│   ├── server_app.py                   # Flower ServerApp (SpeechLLMFedAvg)
│   ├── trainer.py                      # SpeechLLMLightning (WavLM + TinyLlama)
│   ├── trainer_voxtral.py              # VoxtralLightning (Voxtral end-to-end)
│   ├── dataset.py                      # Dataset for speech-llm
│   ├── dataset_voxtral.py              # Dataset for Voxtral
│   ├── evaluate_fl_model.py            # Checkpoint evaluation (both models)
│   ├── model/
│   │   ├── encoder.py                  # WavLM / Whisper encoder wrappers
│   │   ├── connector.py                # Linear / LinearPool / CNN connectors
│   │   ├── llm.py                      # LLM + LoRA loader
│   │   └── voxtral.py                  # Voxtral model loader + LoRA
│   ├── prepare_mls_fl.py               # MLS dataset download + preparation
│   ├── create_experiment_partitions.py  # FL partition generator
│   └── run_experiments.sh              # Batch experiment runner
├── deploy/
│   ├── flower_speech_llm.def           # Singularity definition
│   ├── Dockerfile                      # Docker image
│   └── run_bsc.sh                      # SLURM job script for BSC HPC
├── pyproject.toml                      # Project + Flower config
└── README.md
```

---

## Credits

- **Original app**: [`@mnabih/speech_llm_fl`](https://flower.ai/apps/mnabih/speech_llm_fl/) by **Mohamed Nabih**
- **Flower Framework**: [flower.ai](https://flower.ai/)
- **WavLM**: [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large)
- **TinyLlama**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Voxtral**: [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- **LoRA / PEFT**: [huggingface/peft](https://github.com/huggingface/peft)
- **PyTorch Lightning**: [lightning.ai](https://lightning.ai/docs/pytorch/stable/)

---

## License

Apache License 2.0