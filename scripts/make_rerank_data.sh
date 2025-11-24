#!/bin/bash
#SBATCH --job-name=rerank_data
#SBATCH --output=logs/rerank_data_%j.out
#SBATCH --error=logs/rerank_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-preempt
#SBATCH --gres=cpu:0
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=oyilmazel@umass.edu

module load conda/latest
module load cuda/12.6

nvidia-smi
lscpu
conda activate lenv

export OPENBLAS_NUM_THREADS=1

python fast_rerank_dataset.py
