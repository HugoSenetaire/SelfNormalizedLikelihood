
from .EnergyForDistribution import (
    get_ConvEnergy,
    get_EnergyCategoricalDistrib,
    get_EnergyIsing,
    get_EnergyPoissonDistribution,
    get_EnergyRBM,
    get_fc_energy,
)

dic_energy = {
    "fc": get_fc_energy,
    "conv": get_ConvEnergy,
    "rbm": get_EnergyRBM,
    "categorical": get_EnergyCategoricalDistrib,
    "poisson": get_EnergyPoissonDistribution,
    "ising": get_EnergyIsing,
}

def get_energy(input_size, cfg):
    if cfg.energy.energy_name is None :
        raise ValueError("Energy name not specified")
    if cfg.energy.energy_name not in dic_energy.keys():
        raise ValueError("Energy name not valid")

    energy = dic_energy[cfg.energy.energy_name]
    energy = energy(input_size=input_size, cfg = cfg.energy)
    return energy


from .ExplicitBiasForDistribution import MockBias, ScalarBias


def get_explicit_bias(cfg):
    if cfg.explicit_bias is None or cfg.explicit_bias.explicit_bias_name is None :
        explicit_bias = MockBias()
    elif cfg.explicit_bias.explicit_bias_name == 'scalar' :
        explicit_bias = ScalarBias()
    else :
        raise ValueError('Explicit bias name not valid')
    return explicit_bias