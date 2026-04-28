#!/bin/bash
#SBATCH --job-name=flower_speech_llm_centralized
#SBATCH --output=./slurm_logs/flower_speech_llm_centralized_%j.out
#SBATCH --error=./slurm_logs/flower_speech_llm_centralized_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "QoS: $SLURM_JOB_QOS"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "=========================================="

# Load required modules for MN5
module purge
module load singularity

# Define paths (adjust to your actual paths)
#PATH_SINGULARITY="/gpfs/projects/ehpc628/aleixsant/singularity_containers/ds_nvidia_crossdev_FedEloquence-dev"
PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
#CACHE_DIR="/gpfs/scratch/ehpc628/ehpc628XXX/transformers_cache"
CACHE_DIR='/gpfs/scratch/ehpc628/models/'
TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"
REPO_DIR="/opt/flower_speech_llm"

# Create necessary directories
mkdir -p ./slurm_logs
mkdir -p $CACHE_DIR
mkdir -p $TORCH_EXT_DIR

# MN5 H100-specific optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0

# Export command to run inside singularity
export CMD="
# Set environment variables
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TORCH_EXTENSIONS_DIR=$TORCH_EXT_DIR
# export TRANSFORMERS_CACHE=$CACHE_DIR
# export HF_DATASETS_CACHE=$CACHE_DIR
export DS_BUILD_OPS=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
# Unset SLURM env vars so Lightning uses its own DDP subprocess spawning
# instead of expecting srun-managed processes (ntasks=1 vs devices=4 conflict)
for v in \$(env | grep ^SLURM_ | cut -d= -f1); do unset \$v; done
# CUDA optimizations for H100
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Display GPU info
nvidia-smi
echo 'CUDA_VISIBLE_DEVICES: '\$CUDA_VISIBLE_DEVICES

cd $REPO_DIR
. /opt/venv/bin/activate
python -m flower_speech_llm.train_centralized --config configs/centralized.yaml

echo 'Experiment completed successfully'
"

echo "Starting Singularity container on MN5 acceleration partition..."
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Sandbox directory: $SANDBOX_DIR"
echo ""

# Run singularity with H100 GPU support
singularity exec \
    --nv \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    --bind /gpfs:/gpfs \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$CMD"

#EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE