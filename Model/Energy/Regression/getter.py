
from .EnergyForRegression import get_EnergyNetworkRegression_Large, get_EnergyNetworkRegression_Toy
from .FeatureExtractor import get_Resnet18_FeatureExtractor, get_ToyFeatureNet, MockFeatureExtractor

import numpy as np


dic_energy_regression = {
    'fc': get_EnergyNetworkRegression_Large,
    'toy': get_EnergyNetworkRegression_Toy,
}


def get_energy_regression(input_size_x_feature, input_size_y, cfg):
    if cfg.energy.energy_name is None :
        raise ValueError("Energy name not specified")
    if cfg.energy.energy_name not in dic_energy_regression.keys():
        raise ValueError("Energy name not valid")

    energy = dic_energy_regression[cfg.energy.energy_name]
    energy = energy(input_size_x_feature=input_size_x_feature, input_size_y=input_size_y, cfg=cfg.energy)

    return energy


dic_feature_extractor = {
    'resnet' : get_Resnet18_FeatureExtractor,
    'toy' : get_ToyFeatureNet,
}


def get_feature_extractor_regression(
    cfg,
    input_size_x,
):
    # if "feature_extractor_name" not in args_dict:
    if cfg.feature_extractor is None or cfg.feature_extractor.feature_extractor_name is None :
        return MockFeatureExtractor(input_size_x = input_size_x)
    if cfg.feature_extractor.feature_extractor_name not in dic_feature_extractor:
        raise ValueError("Feature extractor name not valid")
    
    feature_extractor = dic_feature_extractor[cfg.feature_extractor.feature_extractor_name]
    feature_extractor = feature_extractor(input_size_x=input_size_x, cfg = cfg.feature_extractor)
    
    
    if not cfg.feature_extractor.train_feature_extractor :
        for param in feature_extractor.parameters():
            param.requires_grad = False
    return feature_extractor



from .ExplicitBiasForRegression import get_Layer1FC, get_Layer2FC, get_Layer3FC, MockBiasRegression
dic_explicit_bias_regression = {
    '1_layer_fc' : get_Layer1FC,
    '2_layer_fc' : get_Layer2FC,
    '3_layer_fc' : get_Layer3FC,
}


def get_explicit_bias_regression(cfg,
                      input_size_x_feature,
                      ):
    if cfg.explicit_bias.explicit_bias_name not in dic_explicit_bias_regression:
        raise ValueError('Explicit bias name not valid')
    if cfg.explicit_bias is None or cfg.explicit_bias.explicit_bias_name is None :
            return MockBiasRegression(input_size_x_feature = input_size_x_feature)
    explicit_bias = dic_explicit_bias_regression[cfg.explicit_bias.explicit_bias_name]
    explicit_bias = explicit_bias(input_size_x_feature=input_size_x_feature, cfg = cfg.explicit_bias)
    return explicit_bias