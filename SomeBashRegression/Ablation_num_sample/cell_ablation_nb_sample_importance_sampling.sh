CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_1.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=1 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_1.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


############
CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=2 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_8.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

##########

CUDA_VISIBLE_DEVICES=3 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_16.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=3 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_16.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml

###########


CUDA_VISIBLE_DEVICES=4 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_32.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=4 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_32.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


##########
CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_64.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_64.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


##########


CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_128.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/proposal.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml


CUDA_VISIBLE_DEVICES=5 python main_trainer_regression.py --output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor--output_folder '/scratch/hhjs/selfnormalized/Results_trainfeatureextractor' --train_feature_extractor  \
--yamldataset 'Dataset/MissingDataDataset/YAMLExamples/cell_count.yaml' \
--yamlebm Model/YAMLREGRESSION/YAMLEBM_AblationTrainingSamples/importance_sampling_128.yaml Model/YAMLREGRESSION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLREGRESSION/YAMLBASEDIST/none.yaml \
Model/YAMLREGRESSION/YAMLENERGY/energy_fc_large_bias.yaml Model/YAMLREGRESSION/YAMLOPTIMIZATION/adam1e-3.yaml
