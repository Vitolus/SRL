#!/bin/bash
#SBATCH --job-name=SRL_gemma
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/davide.vitagliano/SRL/gemma_%j.log
#SBATCH --chdir=/home/davide.vitagliano/SRL

# Ensure the user passed at least an action and an SRL type
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch run_gemma.slurm <action> <srl_type> [lang1 lang2 ...]"
    echo "Actions: train | eval"
    echo "Examples:"
    echo "  sbatch run_gemma.slurm train dependency EN-s ZH-s"
    echo "  sbatch run_gemma.slurm eval dependency EN-s ZH-s"
    echo "  sbatch run_gemma.slurm eval dependency"
    exit 1
fi

ACTION=$1
SRL_TYPE=$2
MASTER_PORT=$(shuf -i 29501-29700 -n 1)

# Fallback to 2 if SLURM_GPUS_ON_NODE is unset
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}

# Shift twice to remove action and srl_type from the arguments list
shift 2

# Everything remaining contains your languages and optional flags like --zero-shot
EXTRA_ARGS="$@"
# Conditionally append the --langs flag only if languages were passed
if [ -n "$EXTRA_ARGS" ]; then
    LANG_ARG="--langs $EXTRA_ARGS"
else
    LANG_ARG=""
fi

echo "========================================="
echo "Starting Distributed Job"
echo "Action: $ACTION"
echo "SRL Type: $SRL_TYPE"
echo "Languages: ${EXTRA_ARGS:-"None (Zero-Shot Mode)"}"
echo "Allocated GPUs: $NUM_GPUS"
echo "========================================="

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run using torchrun natively pointing to gemma_finetune.py
python ./src/gemma_finetune.py --action $ACTION --srl-type $SRL_TYPE $LANG_ARG

exit