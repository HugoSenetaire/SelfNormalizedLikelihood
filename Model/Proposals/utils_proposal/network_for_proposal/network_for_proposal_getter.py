from .dc_gan_gen import get_DCGANGenerator
from .mlp_proposal import get_mlp_generator
from .resnet_generator import get_ResnetGenerator

dic_proposal_network = {
    "DCGAN": get_DCGANGenerator,
    "resnet": get_ResnetGenerator,
    "mlp": get_mlp_generator,
}


def get_network_for_proposal(
    input_size,
    cfg,
):
    gen = dic_proposal_network[cfg.network_proposal_name](input_size, cfg)
    return gen
