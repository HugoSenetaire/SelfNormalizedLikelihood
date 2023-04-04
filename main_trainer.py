from default_args import default_args_main, check_args_for_yaml
from Dataset.MissingDataDataset.prepare_data import get_dataset
from Model.Utils.model_getter import get_model
from Model.Utils.dataloader_getter import get_dataloader
from Model.Trainer import dic_trainer
import pytorch_lightning as pl
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

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


    # Get Dataset :
    complete_dataset, complete_masked_dataset = get_dataset(args_dict,)
    train_loader = get_dataloader(complete_masked_dataset.dataset_train, args_dict, shuffle = True)
    val_loader = get_dataloader(complete_masked_dataset.dataset_val, args_dict)
    test_loader = get_dataloader(complete_masked_dataset.dataset_test, args_dict)
    
    

    
    if args_dict['yamldataset'] is not None :
        save_dir = os.path.join(args_dict['output_folder'],os.path.basename(args_dict['yamldataset']).split(".yaml")[0])
    else :
        save_dir = os.path.join(args_dict['output_folder'],args_dict['dataset_name'])

    if args_dict['yamlebm'] is not None and len(args_dict['yamlebm'])>0:
        list_element = args_dict['yamlebm']
        list_element.sort()
        for element in list_element:
            save_dir = os.path.join(save_dir,os.path.basename(element).split(".yaml")[0])
    else :
        save_dir = os.path.join(save_dir,args_dict['ebm_name'])

    

    # Get EBM :
    ebm = get_model(args_dict, complete_dataset, complete_masked_dataset)    



    # Get Trainer :
    algo = dic_trainer[args_dict['trainer_name']](ebm, args_dict)


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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, save_top_k=2, monitor="val_loss")

    # Train :
    trainer = pl.Trainer(accelerator=accelerator,
                        default_root_dir=save_dir,
                        callbacks=[checkpoint_callback],
                        # devices = len(devices),
                        strategy = strategy,
                        precision=16,
                        max_steps= args_dict['max_steps'],
                        resume_from_checkpoint=ckpt_path)
    
    trainer.fit(algo, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # algo = algo.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.checkpoint_callback.best_model_path
    trainer.test(algo, dataloaders=test_loader)

    if np.prod(complete_dataset.get_dim_input()) == 2:
        nx = 1000
        ny = 1000
        x = np.linspace(-3, 3, nx)
        y = np.linspace(-3, 3, ny)
        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        xy = torch.from_numpy(xy).float()
        # xy = xy
        z = (-algo.ebm.energy(xy)).exp().detach().cpu().numpy()
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
        plt.savefig(os.path.join(save_dir, "contour_last.png"))
        plt.close()

    
    # print("Best model path : ", trainer.checkpoint_callback.best_model_path)
    # print(torch.load(trainer.checkpoint_callback.best_model_path).keys())
    # print(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"].keys())
    algo.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])

    if np.prod(complete_dataset.get_dim_input()) == 2:
        nx = 1000
        ny = 1000
        x = np.linspace(-3, 3, nx)
        y = np.linspace(-3, 3, ny)
        xx, yy = np.meshgrid(x, y)
        xy = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        xy = torch.from_numpy(xy).float()
        # xy = xy
        z = (-algo.ebm.energy(xy)).exp().detach().cpu().numpy()
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
        plt.close()

  

        