
python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/swiss_roll.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/checkerboard.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/four_circles.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/crescent_cubed.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml



python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/s_curve.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/moon_dataset.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM/elbo.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml