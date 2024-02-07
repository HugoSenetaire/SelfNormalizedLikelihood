#!/bin/bash	
#SBATCH --job-name=jupyter
#SBATCH --account dd-23-138
#SBATCH --output=jupyter-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

echo "Done: $(date +%F-%R:%S)"