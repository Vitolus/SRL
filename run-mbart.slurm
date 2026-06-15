#!/bin/bash
#SBATCH --job-name=SRL_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home/lorenzo.pastore/SRL/finetune_%j.log
#SBATCH --chdir=/home/lorenzo.pastore/SRL

# Ensure the user passed a model, and action, an SRL type, and at least one language
if [ -z "$1" ] || [ -z "$2" ]  || [ -z "$3" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch run.slurm <action> <srl_type> <lang1> [lang2] ..."
    echo "Actions: train | eval | tune"
    echo "Example: sbatch run.slurm train dependency EN ZH"
    exit 1
fi

ACTION=$1
SRL_TYPE=$2
MASTER_PORT=$(shuf -i 29501-29700 -n 1)
NUM_GPUS=$SLURM_GPUS_ON_NODE

# Shift trice to remove model, action and srl_type from the arguments list
shift 2

# Everything remaining ($@) is just your languages
LANGS="$@"

echo "========================================="
echo "Starting Distributed Job"
echo "Action: $ACTION"
echo "SRL Type: $SRL_TYPE"
echo "Languages: $LANGS"
echo "========================================="

source /home/lorenzo.pastore/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use torchrun to launch the script across # GPUs natively
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT ./src/finetune_mbart.py --action $ACTION --srl-type $SRL_TYPE --langs $LANGS

exit