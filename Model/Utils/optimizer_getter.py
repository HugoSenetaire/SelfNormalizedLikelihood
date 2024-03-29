import itertools

import dadaptation
import torch
import torch.optim.lr_scheduler as lr_scheduler


def _get_adamw(cfg, list_parameters_gen):
    optim = torch.optim.AdamW(
        itertools.chain(*list_parameters_gen),
        lr=cfg.lr,
        betas=(cfg.b1, cfg.b2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    return optim


def get_optimizer(cfg, list_parameters_gen):
    if cfg.optimizer == "adamw":
        return _get_adamw(cfg, list_parameters_gen)
    else:
        raise ValueError("Optimizer name not valid")


def get_scheduler(cfg, optim, feedback_scheduler= [], standard_scheduler= []):
    if (
        cfg is None
        or cfg.scheduler_name is None
        or cfg.scheduler_name == "no_scheduler"
    ):
        feedback_scheduler.append(None)
        standard_scheduler.append(None)
    elif cfg.scheduler_name == "step_lr":
        standard_scheduler.append(lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma))
        feedback_scheduler.append(None)
    elif cfg.scheduler_name == "cyclic_lr":
        standard_scheduler.append(lr_scheduler.CyclicLR(
            optim,
            base_lr=cfg.base_lr,
            max_lr=cfg.max_lr,
            step_size_up=cfg.step_size_up,
            cycle_momentum=False,
        ))
        feedback_scheduler.append(None)
    elif cfg.scheduler_name == "cosine_lr":
        standard_scheduler(lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.T_max, eta_min=cfg.eta_min
        ))
        feedback_scheduler.append(None)
    elif cfg.scheduler_name == "reduce_lr_on_plateau":
        feedback_scheduler.append(lr_scheduler.ReduceLROnPlateau(
            optim,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
            threshold=cfg.threshold,
            threshold_mode=cfg.threshold_mode,
            cooldown=cfg.cooldown,
            min_lr=cfg.min_lr,
            eps=cfg.eps,
            verbose=cfg.verbose,
        ))
        standard_scheduler.append(None)
    else:
        raise ValueError("Scheduler name not valid")


# def get_optimizer(args_dict, list_parameters_gen):
#     """
#     Get optimizer from PyTorch. This function reads a dictionary of parameters
#     and returns the asked optimizer with the right parameters. If no parameters
#     are given in the dictionary, default values from PyTorch are used.

#     Args:
#         cfg (dataclass): a dataclass containing the parameters.

#         list_parameters_gen: a list of parameters generator
#     """
#     # Get the optimizer creation function from torch.optim
#     if "OPTIMIZER" not in args_dict:
#         optim_function = getattr(torch.optim, "Adam")(
#             itertools.chain(*list_parameters_gen), lr=1e-4
#         )
#     else:
#         try:
#             optim_name = args_dict["OPTIMIZER"]["NAME"]
#         except KeyError:
#             optim_name = "Adam"

#         if optim_name in torch.optim.__dict__.keys():
#             optim_function = getattr(torch.optim, optim_name)
#             if "PARAMS" not in args_dict["OPTIMIZER"]:
#                 optim_function = optim_function(
#                     itertools.chain(*list_parameters_gen), lr=1e-4
#                 )
#             else:
#                 optim_function = optim_function(
#                     itertools.chain(*list_parameters_gen),
#                     **args_dict["OPTIMIZER"]["PARAMS"]
#                 )

#         elif optim_name in dadaptation.__dict__.keys():
#             optim_function = getattr(dadaptation, optim_name)
#             if "PARAMS" not in args_dict["OPTIMIZER"]:
#                 optim_function = optim_function(
#                     itertools.chain(*list_parameters_gen), lr=1
#                 )
#             else:
#                 if args_dict["OPTIMIZER"]["PARAMS"]["lr"] != 1:
#                     print(
#                         "Defazio recommends to use a learning rate 1, currently {}".format(
#                             args_dict["OPTIMIZER"]["PARAMS"]["lr"]
#                         )
#                     )
#                 optim_function = optim_function(
#                     itertools.chain(*list_parameters_gen),
#                     **args_dict["OPTIMIZER"]["PARAMS"]
#                 )
#         else:
#             raise ValueError("Optimizer name not valid")

#     return optim_function


# def get_scheduler(args_dict, optim):
#     if "SCHEDULER" not in args_dict:
#         return None
#     else:
#         if "NAME" not in args_dict["SCHEDULER"]:
#             return None
#         else:
#             if "PARAMS" not in args_dict["SCHEDULER"]:
#                 return getattr(
#                     torch.optim.lr_scheduler, args_dict["SCHEDULER"]["NAME"]
#                 )(optim)
#             else:
#                 return getattr(
#                     torch.optim.lr_scheduler, args_dict["SCHEDULER"]["NAME"]
#                 )(optim, **args_dict["SCHEDULER"]["PARAMS"])
