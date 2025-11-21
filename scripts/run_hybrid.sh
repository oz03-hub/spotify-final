#!/bin/bash
#SBATCH --job-name=hyb
#SBATCH --output=logs/hyb_%j.out
#SBATCH --error=logs/hyb_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oyilmazel@umass.edu

module load conda/latest

conda activate retrieval_env

python hybrid_ret.py --retrain --model_dir /scratch4/workspace/oyilmazel_umass_edu-mpd/models/ 
