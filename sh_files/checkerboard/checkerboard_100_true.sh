#!/bin/bash	
# NAME="checkerboard_100_true"
#SBATCH --job-name=checkerboard_100_true
#SBATCH --account dd-23-138
#SBATCH --output=checkerboard_100_true-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
python main_trainer_checkerboard.py ebm.train_ais=True ebm.nb_transitions_ais=100
echo "Done: $(date +%F-%R:%S)"