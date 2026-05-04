#!/bin/bash
#SBATCH --job-name=SRL_hypertune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/SRL/hypertune.log
#SBATCH --chdir=/home/davide.vitagliano/SRL

# Ensure the user passed an SRL type and at least one language
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch run_hyper.slurm <model> <srl_type> <lang1> [lang2] ..."
    echo "Example: sbatch run_hyper.slurm mt5 dependency EN"
    exit 1
fi

MODEL=$1
SRL_TYPE=$2

# Shift twice to remove model and srl_type from the arguments list
shift 2

# Everything remaining ($@) is just your languages
LANGS="$@"

echo "========================================="
echo "Starting Single-GPU Tuning Job"
echo "Action: tune"
echo "Model: $MODEL"
echo "SRL Type: $SRL_TYPE"
echo "Languages: $LANGS"
echo "========================================="

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use standard python execution for a single GPU
python ./src/finetune.py --model $MODEL --action tune --srl-type $SRL_TYPE --langs $LANGS

exit