from argparse import ArgumentParser
from Dataset.MissingDataDataset.default_args import default_args_missingdatadataset
from Model.default_args import default_args_ebm
import yaml
import torch

def default_args_main(parser = None):
    if parser is None :
        parser = ArgumentParser()

    parser = default_args_missingdatadataset(parser = parser, root_default = "./Dataset/Downloaded")
    parser = default_args_ebm(parser = parser)
    
    return parser


def update_args_from_yaml(args_dict, yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_args = yaml.load(f, Loader=yaml.FullLoader)
    args_dict.update(yaml_args)
    return args_dict




def check_args_for_yaml(args_dict):
    for key in list(args_dict.keys()):
        if "yaml" in key:
            if args_dict[key] is not None :
                if isinstance(args_dict[key], list) and len(args_dict[key]) >0:
                    for yaml_file in args_dict[key]:
                        args_dict = update_args_from_yaml(args_dict, yaml_file)
                else : 
                    args_dict = update_args_from_yaml(args_dict, args_dict[key])
    return args_dict