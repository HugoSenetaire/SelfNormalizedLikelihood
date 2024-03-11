import os

import numpy as np
import torch
import wandb
import time
def get_wandb_logger(cfg, my_cfg):
    project_name = f"SNL_{cfg.dataset.dataset_name}"
    experiment_name = f"{cfg.train.trainer_name}_q_{cfg.proposal.proposal_name}_b_{cfg.base_distribution.proposal_name}"
    if cfg.ebm.z_estimator_name == 'ais':
        experiment_name += f"_aisk_{cfg.ebm.nb_step_ais}_aistr_{cfg.ebm.nb_transitions_ais}"
    experiment_name += f"_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if cfg.machine is not None:
            if cfg.machine.machine == "karolina":
                print(f"Working on Karolina's machine, {cfg.machine.wandb_path = }")
                # logger_trainer = WandbLogger(project=f"SNL_{cfg.dataset}", save_dir=cfg.machine.wandb_path, config=my_cfg,)
                logger_trainer = wandb.init(project=project_name, name= experiment_name, config=my_cfg, dir=cfg.machine.wandb_path)
            else:
                print(f"Working on {cfg.machine.machine = }")
                # logger_trainer = WandbLogger(project=project_name, name= experiment_name,config=my_cfg,)
                logger_trainer = wandb.init(project=project_name, name= experiment_name, config=my_cfg)
    else:
        print("You have not specified a machine")
        logger_trainer = wandb.init(project=project_name, name= experiment_name, config=my_cfg)
    return logger_trainer

def find_last_version(dir):
    # Find all the version folders
    list_dir = os.listdir(dir)
    list_dir = [d for d in list_dir if "version_" in d]

    # Find the last version
    last_version = 0
    for d in list_dir:
        version = int(d.split("_")[-1])
        if version > last_version:
            last_version = version
    return last_version


def seed_everything(seed):
    # pl.seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_accelerator(
    cfg,
):
    nb_gpu = torch.cuda.device_count()

    if nb_gpu > 1 and cfg.train.multi_gpu != "ddp":
        raise ValueError("You can only use ddp strategy for multi-gpu training")
    if nb_gpu > 1 and cfg.train.multi_gpu == "ddp":
        strategy = "ddp"
    else:
        strategy = "auto"
    if nb_gpu > 0:
        accelerator = "gpu"
        devices = [k for k in range(nb_gpu)]
    else:
        accelerator = "cpu"
        devices = "auto"
    return nb_gpu, accelerator, devices, strategy


def setup_callbacks(
    cfg,
):
    """
    Setup all the callbacks for the training
    """
    checkpoint_callback_val_log = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, "val_checkpoint"),
        save_top_k=2,
        monitor="val/loss_total",
    )
    checkpoint_callback_val_snl = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, "val_checkpoint"),
        save_top_k=2,
        monitor="val/loss_total_SNL",
    )
    checkpoint_callback_train = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, "train_checkpoint"),
        save_top_k=2,
        monitor="train/loss_total",
    )
    checkpoints = [checkpoint_callback_val_log, checkpoint_callback_val_snl, checkpoint_callback_train]
    if cfg.train.decay_ema is not None and cfg.train.decay_ema > 0:
        ema_callback = EMA(decay=cfg.train.decay_ema)
        checkpoints.append(ema_callback)

    checkpoints.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    return checkpoint_callback_val_snl, checkpoints
