# =============================================================================
# Flower Speech-LLM — Offline HPC Docker Image
# =============================================================================
# Build with internet (e.g. on a login node or workstation):
#   docker build -t flower_speech_llm:latest -f deploy/Dockerfile .
#
# Convert to Singularity SIF (for BSC MareNostrum / CTE-POWER):
#   singularity build flower_speech_llm.sif docker-daemon://flower_speech_llm:latest
#
# All HuggingFace models are pre-downloaded — no internet needed at runtime.
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel AS base

LABEL maintainer="Telefonica-Scientific-Research"
LABEL description="Flower Speech-LLM federated learning (offline HPC image)"
LABEL version="1.0.1"

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# ----- System dependencies -----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        libsndfile1 \
        ffmpeg \
        sox \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----- uv for fast Python package management -----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ----- Python virtual environment -----
RUN uv venv /opt/venv --python 3.12
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# ----- Copy project and install -----
COPY . /opt/flower_speech_llm
WORKDIR /opt/flower_speech_llm

RUN . /opt/venv/bin/activate && uv pip install -e . \
    && uv pip install soundfile librosa

# ----- Pre-download ALL HuggingFace models (offline HPC) -----
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache

RUN . /opt/venv/bin/activate && python -c "\
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer, AutoModelForCausalLM; \
print('Downloading microsoft/wavlm-large...'); \
AutoModel.from_pretrained('microsoft/wavlm-large'); \
print('Downloading microsoft/wavlm-base...'); \
AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base'); \
print('Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0...'); \
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
print('Downloading mistralai/Voxtral-Mini-3B-2507...'); \
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor; \
VoxtralProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507'); \
VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507'); \
print('All models downloaded.'); \
"

# ----- Runtime environment (offline mode) -----
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV WANDB_MODE=offline
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# ----- Verification -----
RUN . /opt/venv/bin/activate && python -c "\
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); \
import flwr; print(f'Flower {flwr.__version__}'); \
import pytorch_lightning as pl; print(f'PL {pl.__version__}'); \
import transformers; print(f'Transformers {transformers.__version__}'); \
import peft; print(f'PEFT {peft.__version__}'); \
from flower_speech_llm.trainer import SpeechLLMLightning; print('speech-llm OK'); \
from flower_speech_llm.model.voxtral import get_voxtral; print('voxtral OK'); \
" && du -sh /opt/hf_cache

ENTRYPOINT ["/opt/venv/bin/flwr"]
CMD ["--help"]
