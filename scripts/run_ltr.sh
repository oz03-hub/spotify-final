#!/bin/bash
#SBATCH --job-name=ltr
#SBATCH --output=logs/ltr_%j.out
#SBATCH --error=logs/ltr_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1

module load conda/latest
conda activate retrieval_env

python ltr.py
