
python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml Model/YAMLDISTRIBUTION/YAMLEBM_Images/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/kde_gaussian_adaptive_image.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml

CUDA_VISIBLE_DEVICES=7 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml Model/YAMLDISTRIBUTION/YAMLEBM_2D/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_adaptive.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml  Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM_Images/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_20.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLEBM_Images/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_20.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_conv_no_bias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/mnist_logit.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml Model/YAMLDISTRIBUTION/YAMLEBM_Images/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_conv_no_bias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/utk_face.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml Model/YAMLDISTRIBUTION/YAMLEBM_Images/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_20.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_large_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml Model/YAMLDISTRIBUTION/YAMLSAMPLER/nuts_image.yaml
