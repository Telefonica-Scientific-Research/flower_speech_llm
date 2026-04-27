# =============================================================================
# Flower Speech-LLM — Offline HPC Docker Image
# =============================================================================
# Build with internet (e.g. on a login node or workstation):
#   docker build -t flower_speech_llm:latest -f deploy/Dockerfile .
#
# Convert to Singularity SIF (for BSC MareNostrum / CTE-POWER):
#   singularity build flower_speech_llm.sif docker-daemon://flower_speech_llm:latest
# or from a saved tar (if docker build is done on premises and SIF build on HPC):
# docker save flower_speech_llm:latest -o flower_speech_llm.tar 
# singularity build --fakeroot flower_speech_llm.sif docker-archive://flower_speech_llm.tar
#
# All HuggingFace models are pre-downloaded — no internet needed at runtime.
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

LABEL maintainer="Telefonica-Scientific-Research"
LABEL description="Flower Speech-LLM federated learning (offline HPC image)"
LABEL version="1.0.2"

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# ----- System dependencies -----
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        build-essential \
        gcc \
        dirmngr \
        ca-certificates \
        git \
        vim \
        pdsh \
        python3-socks \
        libsndfile1 \
        ffmpeg \
        sox \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----- Python 3.12 from deadsnakes (container-local) -----
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----- uv for fast Python package management -----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ----- Python venv — pinned to container-local /usr/bin/python3.12 -----
RUN uv venv /opt/venv --python /usr/bin/python3.12
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# ----- Clone project and install -----
RUN git clone https://github.com/Telefonica-Scientific-Research/flower_speech_llm.git /opt/flower_speech_llm
WORKDIR /opt/flower_speech_llm

RUN . /opt/venv/bin/activate \
    && uv pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install -e . \
    && uv pip install soundfile librosa \
    && uv pip install pysocks "httpx[socks]"

# Patch CVE-2025-32434 torch version check in transformers.
# cu121 only has torch<=2.5.1; all HF models use safetensors format
# so the torch.load vulnerability does not apply.
RUN . /opt/venv/bin/activate \
    && IMPORT_UTILS=$(python -c "import transformers.utils.import_utils; print(transformers.utils.import_utils.__file__)") \
    && sed -i '/def check_torch_load_is_safe/a\    return  # Patched: cu121 torch 2.5.1, models use safetensors' "$IMPORT_UTILS"

## One way is to pre-download the HuggingFace models here 
## (but they are very large, so we skip this step and 
## do it on the HPC login node by rsync or scp from premises instead)
## We keep all models in /gpfs/scratch/ehpc628/models 
## scratch and project folders are automatically bind-mount 
## in the container at runtime.

ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache
ENV HF_HUB_CACHE=/gpfs/scratch/ehpc628/models

RUN mkdir -p /opt/hf_cache && mkdir -p /gpfs/scratch && mkdir -p /gpfs/projects
# ----- Pre-download ALL HuggingFace models (offline HPC) -----
# NOTE: Disabled — models are too large to bake into the image.
# Pre-download them on the HPC login node or bind-mount a cache dir instead.
# RUN . /opt/venv/bin/activate && python -c "\
# from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer, AutoModelForCausalLM; \
# print('Downloading microsoft/wavlm-large...'); \
# AutoModel.from_pretrained('microsoft/wavlm-large'); \
# print('Downloading microsoft/wavlm-base...'); \
# AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base'); \
# print('Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0...'); \
# AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
# AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
# print('Downloading mistralai/Voxtral-Mini-3B-2507...'); \
# from transformers import VoxtralForConditionalGeneration, VoxtralProcessor; \
# VoxtralProcessor.from_pretrained('mistralai/Voxtral-Mini-3B-2507'); \
# VoxtralForConditionalGeneration.from_pretrained('mistralai/Voxtral-Mini-3B-2507'); \
# print('All models downloaded.'); \
# "

# ----- Runtime environment (offline mode) -----
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV WANDB_MODE=offline
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# ----- Verification -----
RUN . /opt/venv/bin/activate && python -c "\
import sys; print(f'Python: {sys.executable} ({sys.version})'); \
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); \
import flwr; print(f'Flower {flwr.__version__}'); \
import pytorch_lightning as pl; print(f'PL {pl.__version__}'); \
import transformers; print(f'Transformers {transformers.__version__}'); \
import peft; print(f'PEFT {peft.__version__}'); \
import socks; print('pysocks OK'); \
from flower_speech_llm.trainer import SpeechLLMLightning; print('speech-llm OK'); \
from flower_speech_llm.model.voxtral import get_voxtral; print('voxtral OK'); \
" && du -sh /opt/hf_cache

ENTRYPOINT ["/opt/venv/bin/flwr"]
CMD ["--help"]
