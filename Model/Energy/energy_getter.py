from .EnergyForDistribution import (
    ConvEnergy,
    EnergyCategoricalDistrib,
    EnergyIsing,
    EnergyPoissonDistribution,
    EnergyRBM,
    fc_energy,
)
from .EnergyForRegression import EnergyNetworkRegression
from .FeatureExtractor import resnet_extractor

dic_energy = {
    "fc": fc_energy,
    "conv": ConvEnergy,
    "rbm": EnergyRBM,
    "categorical": EnergyCategoricalDistrib,
    "poisson": EnergyPoissonDistribution,
    "ising": EnergyIsing,
}

dic_energy_regression = {
    "fc": EnergyNetworkRegression,
}


def get_energy(input_size, args_dict):
    if "energy_name" not in args_dict:
        raise ValueError("Energy name not specified")
    if args_dict["energy_name"] not in dic_energy:
        raise ValueError("Energy name not valid")

    energy = dic_energy[args_dict["energy_name"]]
    if "energy_params" not in args_dict:
        energy = energy(input_size=input_size)
    else:
        energy = energy(input_size=input_size, **args_dict["energy_params"])
    return energy


def get_energy_regression(input_size_x, input_size_y, args_dict):
    if "energy_name" not in args_dict:
        raise ValueError("Energy name not specified")
    if args_dict["energy_name"] not in dic_energy_regression:
        raise ValueError("Energy name not valid")

    energy = dic_energy_regression[args_dict["energy_name"]]
    if "energy_params" not in args_dict:
        args_dict["energy_params"] = {}
    energy = energy(
        input_dim_x=input_size_x, input_dim_y=input_size_y, **args_dict["energy_params"]
    )

    return energy


dic_feature_extractor = {
    "resnet": resnet_extractor,
}


def get_feature_extractor(
    args_dict,
    input_size_x,
):
    if "feature_extractor_name" not in args_dict:
        return None
    if args_dict["feature_extractor_name"] not in dic_feature_extractor:
        raise ValueError("Feature extractor name not valid")

    feature_extractor = dic_feature_extractor[args_dict["feature_extractor_name"]]
    if "feature_extractor_params" not in args_dict.keys():
        args_dict["feature_extractor_params"] = {}
    feature_extractor = feature_extractor(
        input_size=input_size_x, **args_dict["feature_extractor_params"]
    )
    return feature_extractor
