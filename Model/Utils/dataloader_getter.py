import torch


def get_dataloader(dataset, args_dict, shuffle=False):
    if args_dict["dataloader_name"] == "default":
        print(f"batch_size : {args_dict['batch_size']}")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args_dict["batch_size"],
            shuffle=shuffle,
            num_workers=args_dict["num_workers"],
        )
    else:
        raise NotImplementedError
