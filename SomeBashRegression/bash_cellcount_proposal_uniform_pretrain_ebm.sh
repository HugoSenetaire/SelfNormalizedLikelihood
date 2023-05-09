# Self normalized test
CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_pretraining' --ebm_pretraining standard_gaussian \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_pretraining' --ebm_pretraining standard_gaussian \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/importance_sampling.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_pretraining' --ebm_pretraining standard_gaussian \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml
