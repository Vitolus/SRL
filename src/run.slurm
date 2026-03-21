#!/bin/bash
#SBATCH --job-name=speed5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=quadro_rtx_5000:1
#SBATCH --ntasks=1
#SBATCH --output=/home/rtripodi/python/langRep/speed5.out
#SBATCH --error=/home/rtripodi/python/langRep/speed5.err
#SBATCH --chdir=/home/rtripodi/python/langRep

echo "SLURM_NODELIST $SLURM_NODELIST"
eval "$(conda shell.bash hook)"

srun /home/rtripodi/.conda/envs/langRep/bin/python /home/rtripodi/python/langRep/speed_2.py

exit