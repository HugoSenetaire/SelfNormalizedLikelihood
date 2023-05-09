CUDA_VISIBLE_DEVICES=7 python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=7 python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/importance_sampling.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml



CUDA_VISIBLE_DEVICES=7 python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml
