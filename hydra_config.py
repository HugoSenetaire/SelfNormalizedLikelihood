import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


@dataclass
class BaseBaseDistributionConfig:
    base_dist_name: str = MISSING
    train_base_dist: bool = MISSING


@dataclass
class BaseDatasetDistributionConfig:
    download: bool = MISSING
    dataset_name: str = MISSING
    missing_mechanism: str = MISSING


@dataclass
class BaseRegressionDatasetConfig:
    pass


@dataclass
class BaseEnergyDistributionConfig:
    energy_name: str = MISSING


@dataclass
class BaseEnergyRegressionConfig:
    energy_name: str = MISSING


@dataclass
class BaseOptimConfig:
    optimizer: str = MISSING


@dataclass
class AdamwConfig(BaseOptimConfig):
    optimizer: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    b1: float = MISSING
    b2: float = MISSING
    eps: float = MISSING


@dataclass
class BaseProposalConfig:
    proposal_name: str = MISSING
    num_sample_proposal: int = MISSING
    num_sample_proposal_val: int = MISSING
    num_sample_proposal_test: int = MISSING
    train_proposal: bool = MISSING
    proposal_loss_name: str = MISSING
    proposal_pretraining: Optional[str] = MISSING

    def __post_init__(self):
        if self.proposal_loss_name not in ["log_prob", "kl", "log_prob_kl"]:
            raise RuntimeError(
                f"proposal_loss_name should be in ['log_prob', 'kl', 'log_prob_kl'] but got {self.proposal_loss_name}"
            )


@dataclass
class BaseTrainConfig:
    trainer_name: str = MISSING
    max_steps: int = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    output_folder: str = MISSING
    load_from_checkpoint: bool = MISSING
    just_test: bool = MISSING
    seed: int = MISSING
    decay_ema: Optional[float] = MISSING
    task_type: str = MISSING

    def __post_init__(self):
        if self.task_type not in ["regression", "distribution estimation"]:
            raise RuntimeError(
                f"task_type should be in ['regression', 'distribution estimation'] but got {self.task_type}"
            )


@dataclass
class Config:
    base_distribution: BaseBaseDistributionConfig = MISSING
    dataset: BaseDatasetDistributionConfig = MISSING
    energy: BaseEnergyDistributionConfig = MISSING
    optim: BaseOptimConfig = MISSING
    proposal: BaseProposalConfig = MISSING
    train: BaseTrainConfig = MISSING


def main():
    cs = ConfigStore.instance()
    cs.store(name="config_name", node=Config)
    cs.store(
        name="proposal_name", group="base_distribution", node=BaseBaseDistributionConfig
    )
    cs.store(
        name="checkerboard_name",
        group="dataset",
        node=BaseDatasetDistributionConfig,
    )
    cs.store(name="adamw_name", group="optim", node=AdamwConfig)
    cs.store(name="gaussian_name", group="proposal", node=BaseProposalConfig)
    cs.store(
        name="distribution_conv_name",
        group="energy",
        node=BaseEnergyDistributionConfig,
    )
    cs.store(name="self_normalized_name", group="train", node=BaseTrainConfig)


if __name__ == "__main__":
    main()
