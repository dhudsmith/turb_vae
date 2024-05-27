#!/bin/bash
#SBATCH --job-name=turbvae
#~SBATCH --partition=wficai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --time=8:00:00

source activate turb
python turb_vae/train2d.py