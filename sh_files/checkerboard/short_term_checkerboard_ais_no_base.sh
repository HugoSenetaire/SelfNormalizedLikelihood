#!/bin/bash	
#SBATCH --job-name=checkerboard_short_term_ais_no_base
#SBATCH --account dd-23-138
#SBATCH --output=checkerboard_short_term_ais_nobase-%J.log
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu
#SBATCH --gpus 1

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

source ~/.bashrc
conda activate /scratch/project/dd-23-138/conda_envs/pytorch_env
python main_trainer_checkerboard.py --multirun \
base_distribution=no_base_distribution \
train=self_normalized_short_term train.start_with_short_term=1000 \
ebm.train_ais=True train.max_epochs=10 \
ebm.nb_transitions_ais=2,10,25,50 ebm.step_size_ais=0.005,0.1,1.0 \
ebm.nb_step_ais=1 ebm.sigma_ais=1.0,0.1,0.01
echo "Done: $(date +%F-%R:%S)"