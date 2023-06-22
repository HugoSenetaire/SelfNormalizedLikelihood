python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 'Model/YAMLREGRESSION/YAMLBASEDIST/normal_diag.yaml'



python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 'Model/YAMLREGRESSION/YAMLBASEDIST/uniform.yaml'



python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/uniform.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 'Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml'

