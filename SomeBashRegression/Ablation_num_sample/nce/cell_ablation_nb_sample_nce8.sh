


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 0  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml



CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 1  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 2  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 4  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml




CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 0  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml



CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 1  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 2  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor --seed 4  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/nce_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

