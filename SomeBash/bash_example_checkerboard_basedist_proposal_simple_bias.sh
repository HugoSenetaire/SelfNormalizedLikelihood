
python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/swiss_roll.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/checkerboard.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/four_circles.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/crescent_cubed.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml



python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/s_curve.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/moon_dataset.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_2D/self_normalized.yaml Model/YAMLPROPOSAL/kde_gaussian_scott.yaml \
Model/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_2d.yaml