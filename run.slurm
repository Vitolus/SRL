#!/bin/bash
#SBATCH --job-name=mt0_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/SRL/mT0.log
#SBATCH --chdir=/home/davide.vitagliano/SRL

# Ensure the user passed a model, and action, an SRL type, and at least one language
if [ -z "$1" ] || [ -z "$2" ]  || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch mt0_finetune.slurm <action> <srl_type> <lang1> [lang2] ..."
    echo "Actions: train | eval"
    echo "Example: sbatch mt0_finetune.slurm train dependency EN ZH"
    exit 1
fi

MODEL=$1
ACTION=$2
SRL_TYPE=$3

# Shift twice to remove action and srl_type from the arguments list
shift 2

# Everything remaining ($@) is just your languages
LANGS="$@"

echo "========================================="
echo "Starting Distributed Job"
echo "Model: $MODEL"
echo "Action: $ACTION"
echo "SRL Type: $SRL_TYPE"
echo "Languages: $LANGS"
echo "========================================="

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use torchrun to launch the script across all 4 GPUs natively
torchrun ./src/finetune_mT0.py --model $MODEL --action $ACTION --srl-type $SRL_TYPE --langs $LANGS

exit