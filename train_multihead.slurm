#!/bin/bash
#SBATCH --job-name=multihead_convlstm
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/venvs/thesis/bin/activate

cd /gpfs/home2/mzdych/thesis

python train_multihead_convlstm.py