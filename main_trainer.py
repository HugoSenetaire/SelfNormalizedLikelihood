import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from Dataset.MissingDataDataset.prepare_data import get_dataset
from default_args import check_args_for_yaml, default_args_main
from Model.Trainer import dic_trainer
from Model.Utils.Callbacks import EMA
from Model.Utils.dataloader_getter import get_dataloader
from Model.Utils.model_getter import get_model
from Model.Utils.plot_utils import plot_energy_2d, plot_images

try:
    from pytorch_lightning.loggers import WandbLogger
except:
    from lighting.pytorch.loggers import WandbLogger

import logging
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))
    my_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = helpers._trigger_post_init(cfg)
    logger.info(os.linesep + pformat(cfg))


if __name__ == "__main__":
    hydra_config.main()
    main()
    raise ValueError("Stop here")
    parser = default_args_main()
    args = parser.parse_args()
    args_dict = vars(args)
    check_args_for_yaml(args_dict)

    if "seed" in args_dict.keys() and args_dict["seed"] is not None:
        pl.seed_everything(args_dict["seed"])
        np.random.seed(args_dict["seed"])
        torch.manual_seed(args_dict["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Get Dataset :
    complete_dataset, complete_masked_dataset = get_dataset(
        args_dict,
    )
    train_loader = get_dataloader(
        complete_masked_dataset.dataset_train, args_dict, shuffle=True
    )
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)

    args_dict["input_size"] = complete_dataset.get_dim_input()

    if args_dict["yamldataset"] is not None:
        name = os.path.basename(args_dict["yamldataset"]).split(".yaml")[0]
        save_dir = os.path.join(
            args_dict["output_folder"],
            os.path.basename(args_dict["yamldataset"]).split(".yaml")[0],
        )
    else:
        name = args_dict["dataset_name"]
        save_dir = os.path.join(args_dict["output_folder"], args_dict["dataset_name"])

    if args_dict["yamlebm"] is not None and len(args_dict["yamlebm"]) > 0:
        list_element = args_dict["yamlebm"]
        list_element.sort()
        for element in list_element:
            name = name + "_" + os.path.basename(element).split(".yaml")[0]
            save_dir = os.path.join(
                save_dir, os.path.basename(element).split(".yaml")[0]
            )
    else:
        name = name + "_" + args_dict["ebm_name"]
        save_dir = os.path.join(save_dir, args_dict["ebm_name"])

    if "seed" in args_dict.keys() and args_dict["seed"] is not None:
        save_dir = os.path.join(save_dir, "seed_{}".format(args_dict["seed"]))

    args_dict["save_dir"] = save_dir

    # Get EBM :
    if args_dict["just_test"]:
        args_dict["ebm_pretraining"] = False
        args_dict["proposal_pretraining"] = False
    ebm = get_model(
        args_dict, complete_dataset, complete_masked_dataset, loader_train=train_loader
    )

    # Get Trainer :
    algo = dic_trainer[args_dict["trainer_name"]](
        ebm=ebm,
        args_dict=args_dict,
        complete_dataset=complete_dataset,
    )

    nb_gpu = torch.cuda.device_count()
    if nb_gpu > 1 and algo.config["MULTIGPU"] != "ddp":
        raise ValueError("You can only use ddp strategy for multi-gpu training")
    if nb_gpu > 1 and algo.config["MULTIGPU"] == "ddp":
        strategy = "ddp"
    else:
        strategy = None
    if nb_gpu > 0:
        accelerator = "gpu"
        devices = [k for k in range(nb_gpu)]
    else:
        accelerator = None
        devices = None

    # accelerator = 'mps'

    if args.load_from_checkpoint or args_dict["just_test"]:
        ckpt_dir = os.path.join(save_dir, "val_checkpoint")
        last_checkpoint = os.listdir(ckpt_dir)[-1]
        ckpt_path = os.path.join(ckpt_dir, last_checkpoint)
        print("Loading from checkpoint : ", ckpt_path)
        assert os.path.exists(ckpt_path), "The checkpoint path does not exist"
        algo.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        ckpt_path = None

    # Checkpoint callback :
    checkpoint_callback_val = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_dir, "val_checkpoint"),
        save_top_k=2,
        monitor="val_loss",
    )
    checkpoint_callback_train = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_dir, "train_checkpoint"),
        save_top_k=2,
        monitor="train_loss",
    )
    checkpoints = [checkpoint_callback_val, checkpoint_callback_train]
    if args_dict["decay_ema"] is not None and args_dict["decay_ema"] > 0:
        ema_callback = EMA(decay=args_dict["decay_ema"])
        checkpoints.append(ema_callback)

    # Train :
    if "max_epoch" in args_dict.keys() and args_dict["max_epoch"] is not None:
        max_steps = args_dict["max_epoch"] * (len(train_loader) + len(val_loader))
        args_dict["max_steps"] = max_steps

    if "val_check_interval" in args_dict.keys():
        val_check_interval = args_dict["val_check_interval"]
    else:
        val_check_interval = None
    trainer = pl.Trainer(
        accelerator=accelerator,
        # logger=logger,
        default_root_dir=save_dir,
        callbacks=checkpoints,
        # devices = len(devices),
        strategy=strategy,
        precision=16,
        max_steps=args_dict["max_steps"],
        resume_from_checkpoint=ckpt_path,
        val_check_interval=val_check_interval,
    )

    if not args_dict["just_test"]:
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
                save_dir=save_dir,
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
                save_dir,
                algo=None,
                transform_back=complete_dataset.transform_back,
                name="samples_best",
                step="",
            )
