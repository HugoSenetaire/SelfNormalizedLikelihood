import logging
from dataclasses import dataclass
from pathlib import Path
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
    input_size: Optional[int] = None
    seed: Optional[int] = None
    dataloader_name: str = "default"
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None


@dataclass
class BaseRegressionDatasetConfig:
    pass


@dataclass
class BaseEnergyDistributionConfig:
    energy_name: str = MISSING
    ebm_pretraining: bool = MISSING


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
class BaseSchedulerConfig:
    scheduler: str = MISSING


@dataclass
class BaseProposalConfig:
    proposal_name: str = MISSING
    num_sample_proposal: int = MISSING
    num_sample_proposal_val: int = MISSING
    num_sample_proposal_test: int = MISSING
    train_proposal: bool = MISSING
    proposal_loss_name: str = MISSING
    proposal_pretraining: bool = MISSING

    def __post_init__(self):
        if self.proposal_loss_name not in ["log_prob", "kl", "log_prob_kl"]:
            raise RuntimeError(
                f"proposal_loss_name should be in ['log_prob', 'kl', 'log_prob_kl'] but got {self.proposal_loss_name}"
            )


@dataclass
class BaseSamplerConfig:
    sampler_name: str = MISSING

    def __post_init__(self):
        if self.sampler_name not in ["no_sampler", "nuts"]:
            raise RuntimeError(
                f"sampler_name should be in ['nuts'] but got {self.sampler_name}"
            )


@dataclass
class NutsConfig(BaseSamplerConfig):
    input_size: Optional[int] = MISSING
    num_chains: int = MISSING
    num_samples: int = MISSING
    warmup_steps: int = MISSING
    thinning: int = MISSING


@dataclass
class BaseTrainConfig:
    trainer_name: str = MISSING
    max_steps: Optional[int] = MISSING
    max_epochs: Optional[int] = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    output_folder: str = MISSING
    load_from_checkpoint: bool = MISSING
    just_test: bool = MISSING
    seed: int = MISSING
    decay_ema: Optional[float] = MISSING
    task_type: str = MISSING
    save_dir: Optional[Path] = None
    multi_gpu: str = MISSING
    val_check_internals: Optional[bool] = MISSING
    save_energy_every: int = MISSING
    samples_every: int = MISSING

    def __post_init__(self):
        if self.task_type not in ["regression", "distribution estimation"]:
            raise RuntimeError(
                f"task_type should be in ['regression', 'distribution estimation'] but got {self.task_type}"
            )
        if self.save_dir is None:
            logger.warning("save_dir is None")

        if self.multi_gpu not in ["single", "ddp"]:
            raise RuntimeError(
                f"multi_gpu should be in ['single', 'ddp'] but got {self.multi_gpu}"
            )

        if self.max_steps is None and self.max_epochs is None:
            raise RuntimeError("max_steps and max_epochs are both None. Please set one")

        if self.max_steps is not None and self.max_epochs is not None:
            raise RuntimeError(
                "max_steps and max_epochs are both not None. Please set only one"
            )


@dataclass
class Config:
    base_distribution: BaseBaseDistributionConfig = MISSING
    dataset: BaseDatasetDistributionConfig = MISSING
    energy: BaseEnergyDistributionConfig = MISSING
    optim: BaseOptimConfig = MISSING
    proposal: BaseProposalConfig = MISSING
    train: BaseTrainConfig = MISSING

    def _complete_dataset(self):
        self.dataset.seed = self.train.seed
        self.dataset.batch_size = self.train.batch_size
        self.dataset.num_workers = self.train.num_workers

    def _complete_train(self):
        self.train.save_dir = Path(self.train.output_folder) / self.dataset.dataset_name

    def __post_init__(self):
        self._complete_dataset()
        self._complete_train()

        if self.train.just_test:
            logger.info("Just testing the model, setting pretraining to False")
            self.energy.ebm_pretraining = False
            self.proposal.proposal_pretraining = False


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
    cs.store(name="no_scheduler_name", group="scheduler", node=BaseSchedulerConfig)
    cs.store(name="gaussian_name", group="proposal", node=BaseProposalConfig)
    cs.store(
        name="distribution_conv_name",
        group="energy",
        node=BaseEnergyDistributionConfig,
    )
    cs.store(name="self_normalized_name", group="train", node=BaseTrainConfig)
    cs.store(name="no_sampler_name", group="sampler", node=BaseSamplerConfig)
    cs.store(name="nuts_name", group="sampler", node=NutsConfig)


if __name__ == "__main__":
    main()
