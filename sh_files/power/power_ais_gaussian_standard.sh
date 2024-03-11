#!/bin/bash	
#SBATCH --job-name=power_maf
#SBATCH --account dd-23-138
#SBATCH --output=power_maf-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
python main_trainer_uci.py --multirun dataset=power_maf base_distribution=proposal proposal=gaussian \
ebm.train_ais=True ebm.nb_transitions_ais=10,20,50 \
ebm.step_size_ais=0.05,0.1,0.01 regularization.l2_output=0. \
regularization.l2_grad=0.1,1.0,0.01 train.max_epochs=10 

echo "Done: $(date +%F-%R:%S)"