

CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 0 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 0 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 1 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 1 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml



CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 2 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 2 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml



CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 3 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/self_normalized_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor_pretraining' --train_feature_extractor --seed 3 \
--ebm_pretraining gaussian --proposal_pretraining data \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/headpose_biwi.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_Image/nce_train_proposal_both.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/mdn_k4.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

