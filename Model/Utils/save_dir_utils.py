import torch
import numpy as np
import os
import pytorch_lightning as pl
from .Callbacks import EMA


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
    pl.seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_accelerator(cfg,):
    nb_gpu = torch.cuda.device_count()

    if nb_gpu > 1 and cfg.train.multi_gpu != "ddp":
        raise ValueError("You can only use ddp strategy for multi-gpu training")
    if nb_gpu > 1 and cfg.train.multi_gpu == "ddp":
        strategy = "ddp"
    else:
        strategy = None
    if nb_gpu > 0:
        accelerator = "gpu"
        devices = [k for k in range(nb_gpu)]
    else:
        accelerator = None
        devices = None
    return nb_gpu, accelerator, devices


def setup_callbacks(cfg,):
    '''
    Setup all the callbacks for the training
    '''
    checkpoint_callback_val = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, "val_checkpoint"),
        save_top_k=2,
        monitor="val_loss",
    )
    checkpoint_callback_train = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.train.save_dir, "train_checkpoint"),
        save_top_k=2,
        monitor="train_loss",
    )
    checkpoints = [checkpoint_callback_val, checkpoint_callback_train]
    if cfg.train.decay_ema is not None and cfg.train.decay_ema > 0:
        ema_callback = EMA(decay=cfg.train.decay_ema)
        checkpoints.append(ema_callback)

    return checkpoint_callback_val, checkpoints