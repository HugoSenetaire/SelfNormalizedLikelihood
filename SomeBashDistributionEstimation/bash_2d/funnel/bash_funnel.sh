python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/funnel_2d.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_10.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml
