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
