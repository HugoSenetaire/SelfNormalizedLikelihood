import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from Dataset.MissingDataDataset.prepare_data import get_dataset
from Model.Trainer import dic_trainer
from Model.Utils.Callbacks import EMA
from Model.Utils.dataloader_getter import get_dataloader
from Model.Utils.model_getter_distributionestimation import get_model
from Model.Utils.plot_utils import plot_energy_2d, plot_images
from Model.Utils.save_dir_utils import get_accelerator, seed_everything, setup_callbacks

import wandb
import logging
import os
from dataclasses import asdict
from pprint import pformat

import hydra
from omegaconf import OmegaConf

import helpers
import hydra_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

from tensorboardX import SummaryWriter


@hydra.main(version_base="1.1", config_path="conf", config_name="config_mnist_replay_buffer")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = helpers._trigger_post_init(cfg)
    logger.info(os.linesep + pformat(cfg))

    if cfg.dataset.seed is not None:
        seed_everything(cfg.dataset.seed)

    # Get datasets and dataloaders :
    args_dict = asdict(cfg.dataset)
    complete_dataset, complete_masked_dataset = get_dataset(args_dict,)
    train_loader = get_dataloader(complete_masked_dataset.dataset_train, args_dict, shuffle=True)
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)
    cfg.dataset.input_size = complete_dataset.get_dim_input()

    # name and save_dir will be in cfg
    ebm = get_model(cfg, complete_dataset, complete_masked_dataset, loader_train=train_loader)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        ebm = ebm.to(device)
        cfg.train.device = device
    else:
        device = torch.device("cpu")
        cfg.train.device = device

    if cfg.train.load_from_checkpoint or cfg.train.just_test:
        ckpt_dir = os.path.join(cfg.train.save_dir, "val_checkpoint")
        last_checkpoint = os.listdir(ckpt_dir)[-1]
        ckpt_path = os.path.join(ckpt_dir, last_checkpoint)
        print("Loading from checkpoint : ", ckpt_path)
        assert os.path.exists(ckpt_path), "The checkpoint path does not exist"
        algo.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        ckpt_path = None

   
    if cfg.machine is not None:
        if cfg.machine.machine == "karolina":
            print(f"Working on Karolina's machine, {cfg.machine.wandb_path = }")
            # logger_trainer = WandbLogger(project="SelfNormalizedLikelihood", save_dir=cfg.machine.wandb_path, config=my_cfg,)
            logger_trainer = wandb.init(project="SelfNormalizedLikelihood", config=my_cfg, dir=cfg.machine.wandb_path)
        else:
            print(f"Working on {cfg.machine.machine = }")
            # logger_trainer = WandbLogger(project="SelfNormalizedLikelihood",config=my_cfg,)
            logger_trainer = wandb.init(project="SelfNormalizedLikelihood", config=my_cfg)
    else:
        print("You have not specified a machine")
        logger_trainer = wandb.init(project="SelfNormalizedLikelihood", config=my_cfg)
        
    algo = dic_trainer[cfg.train.trainer_name](
        ebm=ebm,
        cfg=cfg,
        device = device,
        logger=logger_trainer,
        complete_dataset=complete_dataset,
        )



    # Handle training duration :
    if cfg.train.max_epochs is not None:
        max_steps = cfg.train.max_epochs * (len(train_loader) + len(val_loader))
        cfg.train.max_steps = max_steps
    else :
        max_steps = cfg.train.max_steps

    algo.train(max_steps, train_loader, val_loader=val_loader)

    


if __name__ == "__main__":
    hydra_config.store_main()
    main()
