#!/bin/bash
#SBATCH --job-name=mt0_finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/SRL/mt0_train.out
#SBATCH --error=/home/davide.vitagliano/SRL/mt0_train.err
#SBATCH --chdir=/home/davide.vitagliano/SRL

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch
# TODO: setup cli arguments for the script and pass them here
# Use torchrun to launch the script across all 4 GPUs natively
srun torchrun --nproc_per_node=4 ./src/finetune_mt0.py --langs EN ZH --srl_type dependency

exit