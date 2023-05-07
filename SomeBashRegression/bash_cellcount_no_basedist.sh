# Self normalized test
python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_logprob.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

# Log prob test
python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/importance_sampling_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/importance_sampling_train_proposal_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/importance_sampling_train_proposal_logprob.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


# NCE Test
python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_kl.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml 

python main_trainer_regression.py --ebm_pretraining standard_gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_logprob.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

