#SBATCH --job-name=mt0_hypertune
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/davide.vitagliano/SRL/mt0_tune.log
#SBATCH --chdir=/home/davide.vitagliano/SRL

# Ensure the user passed an SRL type and at least one language
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch mt0_tune.slurm <srl_type> <lang1> [lang2] ..."
    echo "Example: sbatch mt0_tune.slurm dependency EN"
    exit 1
fi

SRL_TYPE=$1

# Shift once to remove srl_type from the arguments list
shift 1

# Everything remaining ($@) is just your languages
LANGS="$@"

echo "========================================="
echo "Starting Single-GPU Tuning Job"
echo "Action: tune"
echo "SRL Type: $SRL_TYPE"
echo "Languages: $LANGS"
echo "========================================="

source /home/davide.vitagliano/miniforge3/etc/profile.d/conda.sh
conda activate pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use standard python execution for a single GPU
srun python ./src/finetune_mT0.py --action tune --srl-type $SRL_TYPE --langs $LANGS

exit