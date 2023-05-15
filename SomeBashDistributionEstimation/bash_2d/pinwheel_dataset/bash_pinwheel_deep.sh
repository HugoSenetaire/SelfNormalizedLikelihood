



python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/pinwheel.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/pinwheel.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml






python main_trainer.py --proposal_pretraining data \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/pinwheel.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/student.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml