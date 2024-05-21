#!/bin/bash
#SBATCH --job-name=turbvae
#SBATCH --partition=wficai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00

# Your code goes here
source activate turb
python turb_vae/train2d.py