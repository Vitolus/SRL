#!/bin/bash
#SBATCH --job-name=mt0_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/SRL/mt0_train.out
#SBATCH --error=/home/davide.vitagliano/SRL/mt0_train.err
#SBATCH --chdir=/home/davide.vitagliano/SRL

# Ensure the user passed at least an SRL type and one language
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch mt0_finetune.slurm <srl_type> <lang1> [lang2] [lang3] ..."
    echo "Example: sbatch mt0_finetune.slurm dependency EN ZH"
    exit 1
fi

# Grab the first argument as the SRL type (e.g., "dependency" or "span")
SRL_TYPE=$1

# This deletes $1, making $2 the new $1, $3 the new $2, etc.
shift

# Everything remaining ($@) is just your languages
LANGS="$@"

echo "========================================="
echo "Starting Distributed Training Job"
echo "SRL Type: $SRL_TYPE"
echo "Languages: $LANGS"
echo "========================================="

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

# Use torchrun to launch the script across all 4 GPUs natively
srun torchrun --nproc_per_node=4 ./src/finetune_mt0.py --langs $LANGS --srl_type $SRL_TYPE

exit