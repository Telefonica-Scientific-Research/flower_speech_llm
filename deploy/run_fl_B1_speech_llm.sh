#!/bin/bash
#SBATCH --job-name=fl_B1_speech_llm
#SBATCH --output=./slurm_logs/fl_B1_speech_llm_%j.out
#SBATCH --error=./slurm_logs/fl_B1_speech_llm_%j.err
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
echo "FL Experiment B1: Non-IID speaker + FedAvg"
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
PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
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
export DS_BUILD_OPS=0
export RAY_TMPDIR=$SCRATCH_DIR
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

# Display GPU info
nvidia-smi
python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available(), \"GPUs:\", torch.cuda.device_count())'

cd $REPO_DIR
. /opt/venv/bin/activate
./run_fl.sh configs/fl_B1_speaker_fedavg.yaml

echo 'Experiment B1 completed successfully'
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

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
