CUDA_VISIBLE_DEVICES=6 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_very_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml



CUDA_VISIBLE_DEVICES=6 python main_trainer.py --proposal_pretraining True \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/student.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_very_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml


CUDA_VISIBLE_DEVICES=6 python main_trainer.py --proposal_pretraining True \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/student.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_very_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml


CUDA_VISIBLE_DEVICES=6 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_very_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml


CUDA_VISIBLE_DEVICES=6 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_very_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml