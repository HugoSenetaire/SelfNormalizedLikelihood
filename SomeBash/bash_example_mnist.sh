
python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_Images/self_normalized.yaml Model/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_Images/self_normalized.yaml Model/YAMLPROPOSAL/gaussian_mixture_20.yaml \
Model/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLBASEDIST/proposal.yaml Model/YAMLEBM_Images/self_normalized.yaml Model/YAMLPROPOSAL/gaussian_mixture_20.yaml \
Model/YAMLENERGY/energy_conv_no_bias.yaml Model/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLBASEDIST/normal_diag.yaml Model/YAMLEBM_Images/self_normalized.yaml Model/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLENERGY/energy_conv_no_bias.yaml Model/YAMLOPTIMIZATION/adam1e-4.yaml Model/YAMLSAMPLER/nuts_image.yaml