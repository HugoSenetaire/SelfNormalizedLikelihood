CUDA_VISIBLE_DEVICES=1 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/ising.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized_ising.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/ising.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_ising.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

CUDA_VISIBLE_DEVICES=2 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/ising.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized_ising_verylargesample.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/ising.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_ising.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml