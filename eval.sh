#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=all
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00         # Set expected wall time
#SBATCH --job-name="eval"
#SBATCH --output="eval.out"

# Get k1 and k2 from command line arguments

# Activate the desired Conda environment
source ~/.bashrc  # Make sure Conda is initialized in your shell
conda activate tf-gpu-jk

module load cuda/11.8

# Update PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/pystrum
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmorph
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite-sandbox
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmorph-sandbox

# Start Jupyter notebook with dynamic k1 and k2 values
python eval.py 
