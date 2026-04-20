#!/bin/bash
#SBATCH --job-name=flower_speech_llm
#SBATCH --output=logs/flower_slm_%j.out
#SBATCH --error=logs/flower_slm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --exclusive
# =============================================================================
# Flower Speech-LLM — BSC HPC SLURM Job Script
# =============================================================================
#
# Prerequisites:
#   1. Build the SIF image (on a node with internet):
#      singularity build flower_speech_llm.sif deploy/flower_speech_llm.def
#      OR: docker build -t flower_speech_llm:latest -f deploy/Dockerfile .
#           singularity build flower_speech_llm.sif docker-daemon://flower_speech_llm:latest
#
#   2. Upload dataset CSVs to $DATA_DIR (see below)
#
#   3. Submit: sbatch deploy/run_bsc.sh
#
# =============================================================================

# ----- Paths (adjust to your BSC project structure) -----
SIF_IMAGE="${HOME}/containers/flower_speech_llm.sif"
PROJECT_DIR="${HOME}/flower_speech_llm"
DATA_DIR="${HOME}/data/fl_partitions"       # Your manually uploaded datasets
RESULTS_DIR="${HOME}/results/flower_slm_${SLURM_JOB_ID}"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"

# ----- Dataset partition directories inside DATA_DIR -----
# Adjust these to match your uploaded data:
#   ${DATA_DIR}/fl_A1_mixed_316/       (client_0.csv ... client_315.csv)
#   ${DATA_DIR}/fl_dev_316/            (client_0.csv ... client_315.csv)
TRAIN_DIR="${DATA_DIR}/fl_A1_mixed_316"
DEV_DIR="${DATA_DIR}/fl_dev_316"

# ----- Experiment config -----
MODEL_TYPE="speech-llm"          # "speech-llm" or "voxtral"
NUM_ROUNDS=40
FRACTION_FIT=0.3
LOCAL_EPOCHS=10
TRAIN_BATCH_SIZE=4
NUM_WORKERS=3
LORA_R=8
LORA_ALPHA=16

# ----- Create output dirs -----
mkdir -p logs "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            $(hostname)"
echo "GPUs:            ${SLURM_GPUS_ON_NODE:-4}"
echo "SIF Image:       ${SIF_IMAGE}"
echo "Train dir:       ${TRAIN_DIR}"
echo "Dev dir:         ${DEV_DIR}"
echo "Model type:      ${MODEL_TYPE}"
echo "Rounds:          ${NUM_ROUNDS}"
echo "Results:         ${RESULTS_DIR}"
echo "============================================================"

# ----- Verify GPU visibility -----
singularity exec --nv "${SIF_IMAGE}" nvidia-smi

# ----- Run Flower simulation -----
singularity exec --nv \
    --bind "${DATA_DIR}:/data" \
    --bind "${RESULTS_DIR}:/results" \
    "${SIF_IMAGE}" \
    /opt/venv/bin/flwr run /opt/flower_speech_llm \
        --run-config "\
model-type=\"${MODEL_TYPE}\" \
num-server-rounds=${NUM_ROUNDS} \
fraction-fit=${FRACTION_FIT} \
local-epochs=${LOCAL_EPOCHS} \
train-batch-size=${TRAIN_BATCH_SIZE} \
num-workers=${NUM_WORKERS} \
lora-r=${LORA_R} \
lora-alpha=${LORA_ALPHA} \
csv-train-dir=\"/data/fl_A1_mixed_316\" \
csv-dev-dir=\"/data/fl_dev_316\" \
checkpoint-dir=\"/results/checkpoints\" \
"

echo "============================================================"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "============================================================"
