#!/bin/bash
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=2GB
#SBATCH --time=01:00:00

#Purge any possible existing models
module purge

# start environments and load existing modules
module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.7.2-gfbf-2023a
module load scikit-learn/1.3.1-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# run your Python main code here
python3 GANown.py
