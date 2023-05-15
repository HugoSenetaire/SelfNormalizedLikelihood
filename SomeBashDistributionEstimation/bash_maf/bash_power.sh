python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/power_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml