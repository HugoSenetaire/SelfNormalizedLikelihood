

from .normal import Normal

def get_base_dist(args_dict, proposal, input_size):
    if args_dict['base_dist_name'] is not None and args_dict['base_dist_name']!= 'none':
        if not 'base_dist_parameters' in args_dict.keys():
            args_dict['base_dist_parameters'] = {}

        if args_dict['base_dist_name'] == 'Normal':
            base_dist = Normal(dim = input_size, **args_dict['base_dist_parameters'])
        elif args_dict['base_dist_name'] == 'proposal':
            base_dist = proposal
        else:
            raise ValueError("Base distribution not recognized")
    else :
        base_dist = None
    
    return base_dist