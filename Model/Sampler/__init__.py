import logging

from .hmc_grathwohl import *
from .Langevin import (
    LangevinSampler,
    LatentLangevinSampler,
    MetropolisAdjustedLangevinSampler,
    langevin_mala_sample,
    langevin_mala_step,
    langevin_sample,
    langevin_step,
)
from .nuts import NutsSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def get_sampler(cfg):
    liste_name = [
        "sampler_init_proposal",
        "sampler_init_data",
        "sampler_init_buffer",
        "sampler_init_base_dist",
        "sampler_latent_langevin",
    ]
    dic_sampler = {}
    for name in liste_name:
        cfg_sampler = getattr(cfg, name)
        if cfg_sampler is None or cfg_sampler.sampler_name is None:
            logger.warning("No sampler specified, using ")
            dic_sampler[name] = None

        elif cfg_sampler.sampler_name == "langevin":
            dic_sampler[name] = LangevinSampler(
                input_size=cfg.dataset.input_size,
                num_chains=cfg_sampler.num_chains,
                num_samples=cfg_sampler.num_samples,
                warmup_steps=cfg_sampler.warmup_steps,
                thinning=cfg_sampler.thinning,
                step_size=cfg_sampler.step_size,
                sigma=cfg_sampler.sigma,
                clip_max_norm=cfg_sampler.clip_max_norm,
                clip_max_value=cfg_sampler.clip_max_value,
                clamp_min=cfg_sampler.clamp_min,
                clamp_max=cfg_sampler.clamp_max,
            )
        elif cfg_sampler.sampler_name == "latent_langevin":
            dic_sampler[name] = LatentLangevinSampler(
                input_size=cfg.dataset.input_size,
                num_chains=cfg_sampler.num_chains,
                num_samples=cfg_sampler.num_samples,
                warmup_steps=cfg_sampler.warmup_steps,
                thinning=cfg_sampler.thinning,
                step_size=cfg_sampler.step_size,
                sigma=cfg_sampler.sigma,
                clip_max_norm=cfg_sampler.clip_max_norm,
                clip_max_value=cfg_sampler.clip_max_value,
                clamp_min=cfg_sampler.clamp_min,
                clamp_max=cfg_sampler.clamp_max,
            )

        elif cfg_sampler.sampler_name == "mala":
            dic_sampler[name] = MetropolisAdjustedLangevinSampler(
                input_size=cfg.dataset.input_size,
                num_chains=cfg_sampler.num_chains,
                num_samples=cfg_sampler.num_samples,
                warmup_steps=cfg_sampler.warmup_steps,
                thinning=cfg_sampler.thinning,
                step_size=cfg_sampler.step_size,
                sigma=cfg_sampler.sigma,
                clip_max_norm=cfg_sampler.clip_max_norm,
                clip_max_value=cfg_sampler.clip_max_value,
                clamp_min=cfg_sampler.clamp_min,
                clamp_max=cfg_sampler.clamp_max,
            )

        elif cfg_sampler.sampler_name == "nuts":
            dic_sampler[name] = NutsSampler(
                input_size=cfg.dataset.input_size,
                num_chains=cfg_sampler.num_chains,
                num_samples=cfg_sampler.num_samples,
                warmup_steps=cfg_sampler.warmup_steps,
                thinning=cfg_sampler.thinning,
                multiprocess=cfg_sampler.multiprocess,
            )
        else:
            raise NotImplementedError(
                f"Sampler {cfg_sampler.sampler_name} not implemented"
            )

    return dic_sampler
