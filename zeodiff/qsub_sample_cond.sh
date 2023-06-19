#!/bin/sh
#SBATCH -J cond_sample
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --comment pytorch


#module  load  singularity/3.9.7  nvtop/1.1.0  htop/3.0.5

python=/scratch/x2513a08/.conda/envs/zeodiff/bin/python

run=/scratch/x2513a08/ZeoDiff_final/zeodiff/run.py

srun $python $run with train=False n_sample=1000 model_dir='/scratch/x2513a08/ZeoDiff_final/zeodiff/models/' sample_dir='/scratch/x2513a08/ZeoDiff_final/zeodiff/sample_uncond' eval_model='unconditional.ckpt'
