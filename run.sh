#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00         # Set expected wall time
#SBATCH --job-name="k19"
#SBATCH --output="logs.out"

# Get k1 and k2 from command line arguments
k1=$1
k2=$2

# Ensure both arguments are provided
if [ -z "$k1" ] || [ -z "$k2" ]; then
  echo "Usage: $0 <k1> <k2>"
  exit 1
fi

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
python train_fov.py --model gmm --num_dims 192 -lr 1e-6 -k1 $k1 -k2 $k2

