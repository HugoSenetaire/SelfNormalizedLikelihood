import logging

from .hmc_grathwohl import *
from .nuts import NutsSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


def get_sampler(cfg):
    if cfg.sampler is None or cfg.sampler.sampler_name is None:
        logger.warning("No sampler specified, using ")
        return None

    elif cfg.sampler.sampler_name == "nuts":
        return NutsSampler(
            input_size=cfg.dataset.input_size,
            num_chains=cfg.sampler.num_chains,
            num_samples=cfg.sampler.num_samples,
            warmup_steps=cfg.sampler.warmup_steps,
            thinning=cfg.sampler.thinning,
        )
