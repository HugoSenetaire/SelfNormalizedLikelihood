

from .BaseDistDistributionEstimation.normal import Normal
from .BaseDistRegression.normal_regression import NormalRegression

def get_base_dist(args_dict, proposal, input_size, dataset):
    if args_dict['base_dist_name'] is not None and args_dict['base_dist_name']!= 'none':
        if not 'base_dist_parameters' in args_dict.keys():
            args_dict['base_dist_parameters'] = {}

        if args_dict['base_dist_name'] == 'Normal':
            base_dist = Normal(input_size= input_size, dataset=dataset, **args_dict['base_dist_parameters'])
        elif args_dict['base_dist_name'] == 'proposal':
            base_dist = proposal
        else:
            raise ValueError("Base distribution not recognized")
    else :
        base_dist = None
    
    return base_dist

def get_base_dist_regression(args_dict, proposal, input_size_x, input_size_y, dataset):
    if args_dict['base_dist_name'] is not None and args_dict['base_dist_name']!= 'none':
        if not 'base_dist_parameters' in args_dict.keys():
            args_dict['base_dist_parameters'] = {}

        if args_dict['base_dist_name'] == 'Normal':
            base_dist = NormalRegression(input_size_x= input_size_x, input_size_y=input_size_y, dataset=dataset, **args_dict['base_dist_parameters'])
        elif args_dict['base_dist_name'] == 'proposal':
            base_dist = proposal
        else:
            raise ValueError("Base distribution not recognized")
    else :
        base_dist = None
    
    return base_dist