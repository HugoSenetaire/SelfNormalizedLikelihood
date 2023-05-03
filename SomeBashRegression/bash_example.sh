python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml


python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/self_normalized.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml 'Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml'
