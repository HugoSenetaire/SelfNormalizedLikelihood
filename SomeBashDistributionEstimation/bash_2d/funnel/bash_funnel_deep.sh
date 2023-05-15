



python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/funnel_2d.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/funnel_2d.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias_deep.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml





