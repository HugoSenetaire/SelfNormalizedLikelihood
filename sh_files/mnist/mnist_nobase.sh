#!/bin/bash	
#SBATCH --job-name=mnistnobase
#SBATCH --account dd-23-138
#SBATCH --output=mnistnobase-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
python main_trainer.py --multirun ebm.train_ais=True dataset=mnist_logittransformed_paddedto32 \
base_distribution=no_base_distribution \
ebm.nb_transitions_ais=20,50 ebm.step_size_ais=1.0,0.1 ebm.sigma_ais=1.0,0.1,0.01 \
regularization.l2_grad=1.0 proposal.mean=0.0 proposal.std=1.0 max_epochs=20
echo "Done: $(date +%F-%R:%S)"