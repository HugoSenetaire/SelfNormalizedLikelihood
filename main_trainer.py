from default_args import default_args_main, check_args_for_yaml
from Dataset.MissingDataDataset.prepare_data import get_dataset
from Model.Utils.model_getter import get_model
from Model.Utils.dataloader_getter import get_dataloader
from Model.Utils.Callbacks import EMA
from Model.Trainer import dic_trainer
from Model.Sampler import nuts_sampler
import pytorch_lightning as pl
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
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




if __name__ == '__main__' :

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
    complete_dataset, complete_masked_dataset = get_dataset(args_dict,)
    train_loader = get_dataloader(complete_masked_dataset.dataset_train, args_dict, shuffle = True)
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)
    
    args_dict['input_size'] = complete_dataset.get_dim_input()

    
    if args_dict['yamldataset'] is not None :
        name = os.path.basename(args_dict['yamldataset']).split(".yaml")[0]
        save_dir = os.path.join(args_dict['output_folder'],os.path.basename(args_dict['yamldataset']).split(".yaml")[0])
    else :
        name = args_dict['dataset_name']
        save_dir = os.path.join(args_dict['output_folder'],args_dict['dataset_name'])




    if args_dict['yamlebm'] is not None and len(args_dict['yamlebm'])>0:
        list_element = args_dict['yamlebm']
        list_element.sort()
        for element in list_element:
            name = name + "_" + os.path.basename(element).split(".yaml")[0]
            save_dir = os.path.join(save_dir,os.path.basename(element).split(".yaml")[0])
    else :
        name = name + "_" + args_dict['ebm_name']
        save_dir = os.path.join(save_dir,args_dict['ebm_name'])

    args_dict['save_dir'] = save_dir
    # try :
    # logger = WandbLogger(project="ebm", name=name, save_dir=save_dir)    
    # except :
    # logger = SummaryWriter(log_dir=save_dir)


    # Get EBM :
    ebm = get_model(args_dict, complete_dataset, complete_masked_dataset)    



    # Get Trainer :
    algo = dic_trainer[args_dict['trainer_name']](ebm = ebm, args_dict = args_dict, complete_dataset=complete_dataset)


    nb_gpu = torch.cuda.device_count()
    if nb_gpu > 1 and algo.config["MULTIGPU"] != "ddp":
        raise ValueError("You can only use ddp strategy for multi-gpu training")
    if nb_gpu>1 and algo.config["MULTIGPU"] == "ddp":
        strategy = "ddp"
    else :
        strategy = None
    if nb_gpu > 0 :
        accelerator = 'gpu'
        devices = [k for k in range(nb_gpu)]
    else:
        accelerator = None
        devices = None


    
    if args.load_from_checkpoint:
        version_dir = os.path.join(save_dir, "lightning_logs")
        last_version = find_last_version(version_dir)
        ckpt_dir = os.path.join(version_dir, f"version_{last_version}/checkpoints/")
        last_checkpoint = os.listdir(ckpt_dir)[-1]
        ckpt_path = os.path.join(ckpt_dir, last_checkpoint)
        assert os.path.exists(ckpt_path)
    else :
        ckpt_path = None

    # Checkpoint callback :
    checkpoint_callback_val = pl.callbacks.ModelCheckpoint(dirpath=save_dir, save_top_k=2, monitor="val_loss")
    checkpoint_callback_train = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(save_dir,"train_checkpoint"), save_top_k=2, monitor="train_loss")
    checkpoints = [checkpoint_callback_val, checkpoint_callback_train]
    if args_dict['decay_ema'] is not None and args_dict['decay_ema'] > 0:
        ema_callback = EMA(decay = args_dict['decay_ema'])
        checkpoints.append(ema_callback)
    # Train :
    trainer = pl.Trainer(accelerator=accelerator,
                        # logger=logger,
                        default_root_dir=save_dir,
                        callbacks=checkpoints,
                        # devices = len(devices),
                        strategy = strategy,
                        precision=16,
                        max_steps = args_dict['max_steps'],
                        resume_from_checkpoint = ckpt_path)
    
    trainer.fit(algo, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # algo = algo.load_from_checkpoint(checkpoint_callback_val.best_model_path)
    # checkpoint_callback_val.best_model_path
    algo.load_state_dict(torch.load(checkpoint_callback_val.best_model_path)["state_dict"])

    trainer.test(algo, dataloaders=test_loader)
    

    if np.prod(complete_dataset.get_dim_input()) == 2:
        nx = 1000
        ny = 1000
        x = np.linspace(-3, 3, nx)
        y = np.linspace(-3, 3, ny)
        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        xy = torch.from_numpy(xy).float()
        z = (-algo.ebm.calculate_energy(xy)).exp().detach().cpu().numpy()
        z = z.reshape(nx, ny)
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        indexes_to_print = np.random.choice(len(complete_dataset.dataset_train), 10000)
        data = torch.cat([complete_dataset.dataset_train.__getitem__(i)[0] for i in indexes_to_print], dim=0)
        axs[0].scatter(data[:,0], data[:,1], s=1)
        axs[1].contourf(x, y, z, 100)
        axs[2].contourf(x, y, z, 100)
        axs[2].scatter(data[:,0], data[:,1], s=1, color = 'red', alpha = 0.3)
        # Add the colorbar to the figure
        fig.colorbar(axs[1].contourf(x, y, z, 100), ax=axs[1])
        plt.savefig(os.path.join(save_dir, "contour_best.png"))
        algo.logger.log_image(key = "contour_best", images = [os.path.join(save_dir, "contour_best.png")])
        plt.close()

  

        