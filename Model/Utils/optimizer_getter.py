import torch 
def get_optimizer(args_dict, model):
    """
        Get optimizer from PyTorch. This function reads a dictionary of parameters
        and returns the asked optimizer with the right parameters. If no parameters
        are given in the dictionary, default values from PyTorch are used.

        Args:
            args: A dictionary containing optimizer parameters.

            model: a PyTorch model.
    """
    # Get the optimizer creation function from torch.optim
    if "OPTIMIZER" not in args_dict:
        optim_function = getattr(torch.optim, "Adam")(model.parameters(), lr=1e-4)
    else :
        if "NAME" not in args_dict["OPTIMIZER"] and "PARAMS" in args_dict["OPTIMIZER"]:
            optim_function = getattr(torch.optim, "Adam")(**args_dict['OPTIMIZER']['PARAMS'])
        elif "NAME" in args_dict["OPTIMIZER"] and "PARAMS" not in args_dict["OPTIMIZER"]:
            optim_function = getattr(torch.optim, args_dict['OPTIMIZER']['NAME'])(model.parameters(), lr=1e-4)
        else :
            optim_function = getattr(torch.optim, args_dict['OPTIMIZER']['NAME'])(model.parameters(), **args_dict['OPTIMIZER']['PARAMS'])
    return optim_function

def get_scheduler(args_dict, model, optim):
    if "SCHEDULER" not in args_dict:
        return None
    else :
        if "NAME" not in args_dict["SCHEDULER"]:
            return None
        else :
            if "PARAMS" not in args_dict["SCHEDULER"]:
                return getattr(torch.optim.lr_scheduler, args_dict["SCHEDULER"]["NAME"])(optim)
            else :
                return getattr(torch.optim.lr_scheduler, args_dict["SCHEDULER"]["NAME"])(optim, **args_dict["SCHEDULER"]["PARAMS"])