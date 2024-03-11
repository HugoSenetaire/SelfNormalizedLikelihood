#!/bin/bash	
#SBATCH --job-name=mnistexptilting
#SBATCH --account dd-23-138
#SBATCH --output=mnistexptilting-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
python main_trainer.py --multirun ebm.train_ais=True dataset=mnist_paddedto32 \
base_distribution=proposal energy=conv_nijkamp_sn \
ebm.nb_transitions_ais=20,50 ebm.step_size_ais=1.0,0.1,0.01 ebm.sigma_ais=0.03 \
regularization.l2_grad=1.0,0.1 train.max_epochs=20
echo "Done: $(date +%F-%R:%S)"

