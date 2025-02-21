#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:30:00         # Set expected wall time
#SBATCH --job-name="k19"
#SBATCH --output="logs.out"

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



# Start Jupyter notebook
python train_fov.py --model gmm -lr 1e-6 -k1 6 -k2 9
