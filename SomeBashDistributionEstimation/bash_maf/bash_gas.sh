CUDA_VISIBLE_DEVICES=1 python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/proposal.yaml Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/none.yaml Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized.yaml Model/YAMLDISTRIBUTION/YAMLPROPOSAL/standard_gaussian.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml


python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_adaptive.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/gaussian_mixture_5.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_adaptive.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-3.yaml

python main_trainer.py \
--yamldataset Dataset/MissingDataDataset/YAMLExamples/gas_maf.yaml \
--yamlebm Model/YAMLDISTRIBUTION/YAMLBASEDIST/normal_diag.yaml \
Model/YAMLDISTRIBUTION/YAMLMAF/self_normalized_train_proposal.yaml \
Model/YAMLDISTRIBUTION/YAMLPROPOSAL/gaussian_mixture_10.yaml \
Model/YAMLDISTRIBUTION/YAMLENERGY/energy_fc_nobias.yaml \
Model/YAMLDISTRIBUTION/YAMLOPTIMIZATION/adam1e-4.yaml