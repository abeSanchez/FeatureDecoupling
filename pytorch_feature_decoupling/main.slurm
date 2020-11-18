#!/bin/bash
#SBATCH --time=05:15:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=fd-train-slurm-%J.out

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"

NUM_GPUS=8
echo "You were assigned $NUM_GPUS gpu(s)"

# Load modules
module load anaconda/anaconda3
module load gcc/gcc-7.1.0
module load cuda/cuda-10.2

# Activate the GPU version of TensorFlow
source activate py36

# Run TensorFlow
echo
python main.py --exp=TinyImageNet_Decoupling_AlexNet --evaluate 0
echo

# You’re done!
echo "Ending script..."
date