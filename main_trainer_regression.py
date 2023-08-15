import os

import pytorch_lightning as pl
import torch

from Dataset.MissingDataDataset.prepare_data import get_dataset
from Model.Trainer import dic_trainer_regression
from Model.Utils.dataloader_getter import get_dataloader
from Model.Utils.model_getter_regression import get_model_regression
from Model.Utils.save_dir_utils import get_accelerator, seed_everything, setup_callbacks

try:
    from pytorch_lightning.loggers import WandbLogger
except:
    from lighting.pytorch.loggers import WandbLogger

import logging
import os
from dataclasses import asdict
from pprint import pformat

import hydra
from omegaconf import OmegaConf

import helpers
from hydra_config import store_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

from tensorboardX import SummaryWriter


@hydra.main(version_base="1.1", config_path="conf", config_name="config_regression")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = helpers._trigger_post_init(cfg)
    logger.info(os.linesep + pformat(cfg))

    if cfg.dataset.seed is not None:
        seed_everything(cfg.dataset.seed)

    # Get Dataset :
    args_dict = asdict(cfg.dataset)
    complete_dataset, complete_masked_dataset = get_dataset(
        args_dict,
    )
    train_loader = get_dataloader(
        complete_masked_dataset.dataset_train, args_dict, shuffle=True
    )
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)

    cfg.dataset.input_size_x = complete_dataset.get_dim_input()
    cfg.dataset.input_size_y = complete_dataset.get_dim_output()

    # Get EBM :

    ebm = get_model_regression(
        cfg, complete_dataset, complete_masked_dataset, loader_train=train_loader
    )

    # Get Trainer :
    print("Trainer name : ", cfg.train.trainer_name, "\n")
    algo = dic_trainer_regression[cfg.train.trainer_name](
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

    # Train :
    trainer = pl.Trainer(
        accelerator=accelerator,
        default_root_dir=cfg.train.save_dir,
        callbacks=checkpoints,
        strategy=None,
        precision=64,
        max_steps=cfg.train.max_steps,
        resume_from_checkpoint=ckpt_path,
        log_every_n_steps=20,
    )

    #    algo.set_trainer(trainer, test_loader)
    if not cfg.train.just_test:
        trainer.fit(algo, train_dataloaders=train_loader, val_dataloaders=val_loader)
        algo.load_state_dict(
            torch.load(checkpoint_callback_val.best_model_path)["state_dict"]
        )

    trainer.test(algo, dataloaders=test_loader)


if __name__ == "__main__":
    store_main()
    main()
