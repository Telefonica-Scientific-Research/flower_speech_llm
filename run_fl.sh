#!/usr/bin/env bash
# Usage:
#   ./run_fl.sh configs/fl_A1_mixed_fedavg.yaml
#   CUDA_VISIBLE_DEVICES=0,1 ./run_fl.sh configs/fl_B1_speaker_fedavg.yaml
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [extra flwr run args...]"
    echo "Example: CUDA_VISIBLE_DEVICES=0,1 $0 configs/fl_A1_mixed_fedavg.yaml"
    exit 1
fi

# Let Ray manage per-worker CUDA_VISIBLE_DEVICES assignment.
# Do NOT set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES — it prevents
# Ray from assigning individual GPUs to FL client actors.

CONFIG="$1"; shift

# Isolate Flower runtime state (SQLite/token store) per run to avoid
# cross-job lock contention when multiple experiments run concurrently.
if [[ -z "${FLWR_HOME:-}" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    run_id="${SLURM_JOB_ID:-$$}"
    export FLWR_HOME="$PWD/.flwr_runs/flwr_${run_id}_${ts}"
fi
mkdir -p "$FLWR_HOME"

# Create a unique experiment directory under checkpoint-dir to avoid
# collisions when re-running the same YAML (e.g., multiple A1 runs).
EXP_DIR=$(python -c "
import os
from datetime import datetime
import yaml

config_path = '$CONFIG'
cfg = yaml.safe_load(open(config_path)) or {}

base_dir = str(cfg.get('checkpoint-dir', 'FL_SLAM_checkpoints'))
base_dir = os.path.expanduser(os.path.expandvars(base_dir))
config_stem = os.path.splitext(os.path.basename(config_path))[0]
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
job_id = os.environ.get('SLURM_JOB_ID', '').strip()
job_suffix = f'_job{job_id}' if job_id else ''

print(os.path.join(base_dir, f'{config_stem}_{ts}{job_suffix}'))
")

mkdir -p "$EXP_DIR"
cp "$CONFIG" "$EXP_DIR/$(basename "$CONFIG")"

RUN_CONFIG=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
parts = []
for k, v in cfg.items():
    if k.startswith('_'): continue
    if k == 'checkpoint-dir': continue
    if isinstance(v, bool):
        parts.append(f'{k}={str(v).lower()}')
    elif isinstance(v, str):
        parts.append(f'{k}=\"{v}\"')
    else:
        parts.append(f'{k}={v}')
print(' '.join(parts))
")

# Force checkpoint-dir to this run-specific directory.
RUN_CONFIG="$RUN_CONFIG checkpoint-dir=\"$EXP_DIR\""

# Extract simulation options (prefixed with _) for --federation-config
# Also compute num visible GPUs from CUDA_VISIBLE_DEVICES to pass to Ray
FED_CONFIG=$(python -c "
import yaml, os
cfg = yaml.safe_load(open('$CONFIG'))
num_supernodes = cfg.get('_num-supernodes', 316)
num_cpus = cfg.get('_client-num-cpus', 4)
num_gpus = cfg.get('_client-num-gpus', 1.0)
# Count visible GPUs so Ray doesn't claim GPUs outside CUDA_VISIBLE_DEVICES
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if cvd:
    total_gpus = len(cvd.split(','))
else:
    import torch
    total_gpus = torch.cuda.device_count()
parts = [
    f'num_supernodes={num_supernodes}',
    f'client_resources_num_cpus={num_cpus}',
    f'client_resources_num_gpus={num_gpus}',
    f'init_args_num_gpus={total_gpus}',
]
print(' '.join(parts))
")

echo "Config: $CONFIG"
echo "FLWR_HOME: $FLWR_HOME"
echo "Experiment dir: $EXP_DIR"
echo "Run config: $RUN_CONFIG"
echo "Federation config: $FED_CONFIG"
echo ""

printf '%s\n' "$RUN_CONFIG" > "$EXP_DIR/run_config.txt"
printf '%s\n' "$FED_CONFIG" > "$EXP_DIR/federation_config.txt"

flwr run . --run-config "$RUN_CONFIG" --federation-config "$FED_CONFIG" "$@"
