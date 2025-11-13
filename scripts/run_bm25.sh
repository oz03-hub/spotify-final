#!/bin/bash
#SBATCH --job-name=bm25_baseline
#SBATCH --output=logs/bm25_baseline_%j.out
#SBATCH --error=logs/bm25_baseline_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=cpu-preempt
#SBATCH --gres=cpu:0
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1

module load conda/latest
conda activate retrieval_env

python bm25_baseline.py
