python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/self_normalized_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias_no_extractor.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'

python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/nce_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias_no_extractor.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'


python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/eubo_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias_no_extractor.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'


python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/self_normalized_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'

python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/nce_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'


python main_trainer_regression.py \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/1d_regression_1.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_1D/eubo_train_proposal_logprob_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k2.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_small_no_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-2.yaml \
'Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml'
