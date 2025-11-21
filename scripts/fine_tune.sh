#!/bin/bash
#SBATCH --job-name=fine_tune
#SBATCH --output=logs/fine_tune_%j.out
#SBATCH --error=logs/fine_tune_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1

module load conda/latest

conda activate retrieval_env

# CHANGE TO YOUR OWN PATH
python fine_tune.py  --output_dir /scratch4/workspace/oyilmazel_umass_edu-mpd/transformer_finetuned/ 
