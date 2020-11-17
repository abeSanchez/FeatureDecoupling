#!/bin/bash
#SBATCH --time=05:15:00
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --output=cifar10-train-slurm-%J.out

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"

NUM_GPUS=8
echo "You were assigned $NUM_GPUS gpu(s)"

# Load modules
module load anaconda/anaconda3
module load gcc/gcc-7.1.0
module load cuda/cuda-10.2

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
echo

# Activate the GPU version of TensorFlow
source activate pytorch-gpu

# Run TensorFlow
echo
time python models/tutorials/image/cifar10/cifar10_train.py --train_dir=./out --tmp_dir=./tmp
echo

# You’re done!
echo "Ending script..."
date