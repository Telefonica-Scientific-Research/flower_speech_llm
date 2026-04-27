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

RUN_CONFIG=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
parts = []
for k, v in cfg.items():
    if k.startswith('_'): continue
    if isinstance(v, bool):
        parts.append(f'{k}={str(v).lower()}')
    elif isinstance(v, str):
        parts.append(f'{k}=\"{v}\"')
    else:
        parts.append(f'{k}={v}')
print(' '.join(parts))
")

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
echo "Run config: $RUN_CONFIG"
echo "Federation config: $FED_CONFIG"
echo ""

flwr run . --run-config "$RUN_CONFIG" --federation-config "$FED_CONFIG" "$@"
