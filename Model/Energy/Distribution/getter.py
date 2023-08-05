from .EnergyForDistribution import (
    get_BNDC_GAN_Discriminator,
    get_ConvEnergy,
    get_DC_GAN_Discriminator,
    get_DC_GAN_DiscriminatorSN,
    get_EnergyCategoricalDistrib,
    get_EnergyIsing,
    get_EnergyPoissonDistribution,
    get_EnergyRBM,
    get_fc_energy,
    get_ResNetDiscriminator,
)

dic_energy = {
    "fc": get_fc_energy,
    "conv": get_ConvEnergy,
    "rbm": get_EnergyRBM,
    "categorical": get_EnergyCategoricalDistrib,
    "poisson": get_EnergyPoissonDistribution,
    "ising": get_EnergyIsing,
    "dc_gan": get_DC_GAN_Discriminator,
    "dc_gan_sn": get_DC_GAN_DiscriminatorSN,
    "resnet": get_ResNetDiscriminator,
    "bndc_gan": get_BNDC_GAN_Discriminator,
}


def get_energy(input_size, cfg):
    if cfg.energy.energy_name is None:
        raise ValueError("Energy name not specified")
    if cfg.energy.energy_name not in dic_energy.keys():
        raise ValueError("Energy name not valid")

    energy = dic_energy[cfg.energy.energy_name]
    energy = energy(input_size=input_size, cfg=cfg.energy)
    return energy


from .ExplicitBiasForDistribution import MockBias, ScalarBias


def get_explicit_bias(cfg):
    if cfg.explicit_bias is None or cfg.explicit_bias.explicit_bias_name is None:
        explicit_bias = MockBias()
    elif cfg.explicit_bias.explicit_bias_name == "scalar":
        explicit_bias = ScalarBias()
    else:
        raise ValueError("Explicit bias name not valid")
    return explicit_bias
