
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
from Model.Utils.save_dir_utils import seed_everything, get_accelerator, setup_callbacks

try:
    from pytorch_lightning.loggers import WandbLogger
except:
    from lighting.pytorch.loggers import WandbLogger

import logging
from pprint import pformat

import hydra
from omegaconf import OmegaConf
import os
from dataclasses import asdict

import helpers
import hydra_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

from tensorboardX import SummaryWriter



@hydra.main(version_base='1.1', config_path="conf", config_name="config_mnist")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = helpers._trigger_post_init(cfg)
    logger.info(os.linesep + pformat(cfg))

    if cfg.dataset.seed is not None:
        seed_everything(cfg.dataset.seed)


    # Get datasets and dataloaders :
    args_dict = asdict(cfg.dataset)
    complete_dataset, complete_masked_dataset = get_dataset(
        args_dict,
    )
    train_loader = get_dataloader(
        complete_masked_dataset.dataset_train, args_dict, shuffle=True
    )
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)
    cfg.dataset.input_size = complete_dataset.get_dim_input()

    # name and save_dir will be in cfg

    ebm = get_model(cfg, complete_dataset, complete_masked_dataset, loader_train=train_loader)

    algo = dic_trainer[cfg.train.trainer_name](
        ebm=ebm,
        cfg=cfg,
        complete_dataset=complete_dataset,
    )

    nb_gpu, accelerator, strategy = get_accelerator(cfg)

    if cfg.train.load_from_checkpoint or cfg.train.just_test:
        ckpt_dir = os.path.join(cfg.train.save_dir, "val_checkpoint")
        last_checkpoint = os.listdir(ckpt_dir)[-1]
        ckpt_path = os.path.join(ckpt_dir, last_checkpoint)
        print("Loading from checkpoint : ", ckpt_path)
        assert os.path.exists(ckpt_path), "The checkpoint path does not exist"
        algo.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        ckpt_path = None

    # Checkpoint callback :
    checkpoint_callback_val, checkpoints = setup_callbacks(cfg)

    # Handle training duration :
    if cfg.train.max_epochs is not None:
        max_steps = cfg.train.max_epochs * (len(train_loader) + len(val_loader))
        cfg.train.max_steps = max_steps
    val_check_interval = cfg.train.val_check_interval

    # Get Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        default_root_dir=cfg.train.save_dir,
        callbacks=checkpoints,
        strategy=strategy,
        precision=16,
        max_steps=cfg.train.max_steps,
        resume_from_checkpoint=ckpt_path,
        val_check_interval=val_check_interval,
    )

    if not cfg.train.just_test:
        trainer.fit(algo, train_dataloaders=train_loader, val_dataloaders=val_loader)
        algo.load_state_dict(
            torch.load(checkpoint_callback_val.best_model_path)["state_dict"]
        )

    trainer.test(algo, dataloaders=test_loader)
    if algo.sampler is not None:
        if np.prod(complete_dataset.get_dim_input()) == 2:
            samples = algo.samples_mcmc()[0].flatten(1)
            plot_energy_2d(
                algo=algo,
                save_dir=cfg.train.save_dir,
                samples=[algo.example, algo.example_proposal, samples],
                samples_title=[
                    "Samples from dataset",
                    "Samples from proposal",
                    "Samples HMC",
                ],
            )
        else:
            images = algo.samples_mcmc()[0]
            plot_images(
                images,
                cfg.train.save_dir,
                algo=None,
                transform_back=complete_dataset.transform_back,
                name="samples_best",
                step="",
            )


if __name__ == "__main__":
    hydra_config.store_main()
    main()
