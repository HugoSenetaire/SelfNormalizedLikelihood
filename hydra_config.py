import logging
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


@dataclass
class BaseDatasetConfig:
    download: bool = MISSING
    dataset_name: str = MISSING
    dataset_parameters: Optional[dict] = field(default_factory=dict)
    dataloader_name: str = "default"
    batch_size: Optional[int] = 32
    num_workers: Optional[int] = 2
    static_generator_name: Optional[str] = None
    static_generator_parameters: Optional[dict] = None
    dynamic_generator_name: Optional[str] = None
    dynamic_generator_parameters: Optional[dict] = None
    seed: Optional[int] = None
    root: Optional[str] = (
        os.path.dirname(os.path.realpath(__file__)) + "/Dataset/Downloaded"
    )


# defaults_base_energy = [
#     {"dims": [100, 100, 100],},
# ]


@dataclass
class BaseEnergyConfig:
    energy_name: str = MISSING
    ebm_pretraining: Optional[str] = None

    dims: Optional[list] = field(default_factory=lambda: [100, 100, 100])
    activation: Optional[str] = None
    last_layer_bias: Optional[bool] = False

    hidden_dim: Optional[int] = 10

    theta: Optional[float] = 1.0
    learn_theta: Optional[bool] = False

    lambda_: Optional[float] = 1.0
    learn_lambda: Optional[bool] = False

    learn_W: Optional[bool] = False
    learn_b: Optional[bool] = False

    ngf: Optional[int] = 64
    nout: Optional[int] = 1
    weight_norm: Optional[bool] = False


@dataclass
class BaseExplicitBiasConfig:
    explicit_bias_name: Union[str, None] = MISSING
    nb_sample_init_bias: Optional[int] = 1024


@dataclass
class BaseFeatureExtractorConfig:
    feature_extractor_name: Union[str, None] = MISSING
    train_feature_extractor: bool = MISSING
    hidden_dim: Optional[int] = 10


@dataclass
class BaseOptimConfig:
    optimizer: str = MISSING
    clip_grad_norm: Optional[float] = None
    pg_control_data: Optional[float] = 0.0
    pg_control_gen: Optional[float] = 0.0


@dataclass
class AdamwConfig(BaseOptimConfig):
    optimizer: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    b1: float = MISSING
    b2: float = MISSING
    eps: float = MISSING
    pg_control_data: Optional[float] = 0.0
    pg_control_gen: Optional[float] = 0.0
    pg_control_mix: Optional[float] = 0.0
    coef_regul: Optional[float] = 0.0
    clip_grad_norm: Optional[float] = None


@dataclass
class BaseSchedulerConfig:
    scheduler_name: str = MISSING

    step_size: Optional[int] = None
    gamma: Optional[float] = None

    base_lr: Optional[float] = None
    max_lr: Optional[float] = None
    step_size_up: Optional[int] = None
    cycle_momentum: Optional[bool] = True

    T_max: Optional[int] = None
    eta_min: Optional[float] = None

    mode: Optional[str] = "min"  # min, max
    factor: Optional[float] = 0.1
    patience: Optional[int] = 10
    threshold: Optional[float] = 1e-4
    threshold_mode: Optional[str] = "rel"  # rel, abs
    cooldown: Optional[int] = 0
    min_lr: Optional[float] = 0
    eps: Optional[float] = 1e-8
    verbose: Optional[bool] = False
    feedback_loss: Optional[str] = 'SNL' # SNL or IS


@dataclass
class BaseProposalConfig:
    proposal_name: Union[str, None] = MISSING
    covariance_type: Optional[str] = "diag"  # Used in gaussian
    eps: Optional[float] = 1.0e-6  # Used in GaussianMixtureProposal
    n_components: Optional[int] = 10  # Used in GaussianMixtureProposal
    nb_sample_estimate: Optional[int] = 100000
    init_parameters: Optional[str] = "kmeans"
    delta: Optional[float] = 1e-3
    n_iter: Optional[int] = 100
    warm_start: Optional[bool] = False
    fit: Optional[bool] = True

    kernel: Optional[str] = "gaussian"  # Used in KernelDensity
    bandwith: Optional[str] = "scott"  # Used in KernelDensity
    nb_center: Optional[int] = 1000  # Used in KernelDensity

    ranges_std: Optional[list] = field(
        default_factory=lambda: [0.1, 1.0]
    )  # Used in noise gradation adaptive

    lambda_: Optional[float] = 0.1  # Used in Poisson

    mean: Optional[str] = "dataset"  # Used in standard gaussian
    std: Optional[str] = "dataset"  # Used in standard gaussian
    std_multiplier: Optional[
        float
    ] = 1.0  # Used in standard gaussian, just multiply the std to get larger proposal

    K: Optional[int] = 4  # Used in MDN proposal regression

    min_data: Optional[str] = "dataset"  # Used in UNIFORM proposal regression
    max_data: Optional[str] = "dataset"  # Used in UNIFORM proposal regression
    shift_min: Optional[float] = 0.0  # Used in UNIFORM proposal regression
    shift_max: Optional[float] = 0.0  # Used in UNIFORM proposal regression

    real_nvp_num_scales: Optional[int] = 2  # Used in RealNVP proposal
    real_nvp_mid_channels: Optional[int] = 64  # Used in RealNVP proposal
    real_nvp_num_blocks: Optional[int] = 3  # Used in RealNVP proposal
    real_nvp_preprocess: Optional[bool] = False  # Used in RealNVP proposal
    real_nvp_k: Optional[int] = 256  # Used in RealNVP proposal
    epochs: Optional[int] = 10  # Used in RealNVP proposal

    network_proposal_name: Optional[str] = "DCGAN"  # Used in network proposal
    noise_dim: Optional[int] = 2  # Used in vera proposal
    mcmc_lr: Optional[float] = 0.02  # Used in vera hmc proposal
    post_lr: Optional[
        float
    ] = 0.02  # Used in vera proposal, learning rate for learning eta
    init_post_logsigma: Optional[
        float
    ] = 0.1  # Used in vera proposal, initial sigma for learning eta

    activation: Optional[
        Union[str, None]
    ] = None  # Used in network proposal, quite important, depends on where the data is relying
    # If data in [0,1], sigmoid, if data in [-1,1], tanh, else None
    ngf: Optional[int] = 64  # Number of channels after the first conv of DCGAN
    feats: Optional[int] = 128  # Features for the Resnet
    h_dim: Optional[int] = 128  # Hidden dimension for the MLP

    maf_hidden_dim: Optional[list] = field(
        default_factory=lambda: [512]
    )  # Used in MAF proposal
    maf_num_blocks: Optional[int] = 5  # Used in MAF proposal
    maf_use_reverse: Optional[bool] = False  # Used in MAF proposal

    pytorch_flow_name: Optional[str] = "maf"  # Used in pytorch flows proposal
    pytorch_flow_hidden_dim: Optional[int] = 1024  # Used in pytorch flows proposal
    pytorch_flow_num_blocks: Optional[int] = 5  # Used in pytorch flows proposal
    pytorch_flow_act: Optional[str] = "relu"  # Used in pytorch flows proposal


@dataclass
class BaseBaseDistributionConfig(BaseProposalConfig):
    train_base_dist: Optional[bool] = False

    def post_init(self):
        if "adaptive" in self.proposal_name:
            raise RuntimeError(f"Base distribution should not be adaptive")


@dataclass
class BaseProposalTrainingConfig:
    num_sample_train_estimate: Optional[int] = None  # This is used to compare multiple training at once.
    num_sample_proposal: int = MISSING
    num_sample_proposal_val: int = MISSING
    num_sample_proposal_test: int = MISSING
    train_proposal: bool = MISSING
    proposal_loss_name: Union[str, None] = None
    proposal_pretraining: Optional[str] = None
    proposal_pretraining_epochs: Optional[int] = 10
    noise_annealing_init: Optional[float] = 0.0
    noise_annealing_gamma: Optional[float] = 0.9999

    def __post_init__(self):
        if self.train_proposal:
            if self.proposal_loss_name not in ["log_prob", "kl", "log_prob_kl"]:
                raise RuntimeError(
                    f"proposal_loss_name should be in ['log_prob', 'kl', 'log_prob_kl'] but got {self.proposal_loss_name}"
                )
        else:
            if self.proposal_loss_name is not None:
                raise RuntimeError(
                    f"proposal_loss_name should be None but got {self.proposal_loss_name}"
                )


@dataclass
class BaseSamplerConfig:
    sampler_name: str = MISSING

    def __post_init__(self):
        if self.sampler_name not in ["no_sampler", "nuts"]:
            raise RuntimeError(f"sampler_name should be in ['nuts'] but got {self.sampler_name}")


@dataclass
class NutsConfig(BaseSamplerConfig):
    sampler_name: Optional[str] = "nuts"
    num_chains: int = MISSING
    num_samples: int = MISSING
    warmup_steps: int = MISSING
    thinning: int = MISSING
    multiprocess: Optional[bool] = False


@dataclass
class BaseTrainConfig:
    trainer_name: str = MISSING
    max_steps: Optional[int] = MISSING
    max_epochs: Optional[int] = MISSING
    output_folder: str = MISSING
    load_from_checkpoint: bool = MISSING
    just_test: bool = MISSING
    seed: int = MISSING
    decay_ema: Optional[float] = MISSING
    task: str = MISSING
    save_dir: Optional[Path] = None
    multi_gpu: str = MISSING
    val_check_interval: Optional[float] = MISSING
    save_energy_every: int = MISSING
    samples_every: int = MISSING
    sigma: Optional[float] = None
    entropy_weight: Optional[float] = 0.0001
    log_every_n_steps: int = MISSING
    save_locally: Optional[bool] = False
    start_with_IS_until: Optional[Union[None, int]] = 0

    bias_training_iter: Optional[int] = 0
    lr_bias: Optional[float] = 1e-3

    noise_annealing_init: Optional[float] = 0.0
    noise_annealing_gamma: Optional[float] = 0.999

    nb_energy_steps: Optional[int] = 0

    def __post_init__(self):
        if self.task not in ["regression", "distribution_estimation"]:
            raise RuntimeError(f"task should be in ['regression', 'distribution_estimation'] but got {self.task}")
        
        if self.save_dir is None:
            logger.warning("save_dir is None")

        if self.multi_gpu not in ["single", "ddp"]:
            raise RuntimeError(f"multi_gpu should be in ['single', 'ddp'] but got {self.multi_gpu}")

        if self.max_steps is None and self.max_epochs is None:
            raise RuntimeError("max_steps and max_epochs are both None. Please set one")

        if self.max_steps is not None and self.max_epochs is not None:
            raise RuntimeError("max_steps and max_epochs are both not None. Please set only one")

        if "denoising" in self.trainer_name and self.sigma is None:
            raise RuntimeError("Sigma is needed when considering training with denoising models")


@dataclass
class Machine:
    machine: str = MISSING
    wandb_path: Optional[str] = MISSING

    def __post_init__(self):
        if self.machine == "karolina" or self.machine == "dtu_cluser":
            self.wandb_path = pathlib.Path(
                pathlib.Path.home().parent.parent, self.wandb_path
            )
        else:
            self.wandb_path = None


@dataclass
class Config:
    base_distribution: BaseBaseDistributionConfig = MISSING
    dataset: BaseDatasetConfig = MISSING
    energy: BaseEnergyConfig = MISSING
    optim_f_theta: BaseOptimConfig = MISSING
    optim_log_bias: BaseOptimConfig = MISSING
    optim_proposal: BaseOptimConfig = MISSING
    optim_base_dist: BaseOptimConfig = MISSING
    proposal_training: BaseProposalTrainingConfig = MISSING
    proposal: BaseProposalConfig = MISSING
    default_proposal: Optional[Union[BaseProposalConfig, None]] = None
    train: BaseTrainConfig = MISSING
    feature_extractor: Optional[Union[BaseFeatureExtractorConfig, None]] = None
    explicit_bias: BaseExplicitBiasConfig = MISSING
    sampler: Optional[Union[BaseSamplerConfig, None]] = None
    scheduler_f_theta: Optional[Union[BaseSchedulerConfig, None]] = None
    scheduler_log_bias: Optional[Union[BaseSchedulerConfig, None]] = None
    scheduler_proposal: Optional[Union[BaseSchedulerConfig, None]] = None
    scheduler_base_dist: Optional[Union[BaseSchedulerConfig, None]] = None
    machine: Optional[Machine] = None


    def _complete_train(self):
        self.train.save_dir = Path(self.train.output_folder) / self.dataset.dataset_name

    def __post_init__(self):
        # self._complete_dataset()
        self._complete_train()

        if self.train.just_test:
            logger.info("Just testing the model, setting pretraining to False")
            self.energy.ebm_pretraining = None
            self.proposal.proposal_pretraining = None


def store_main():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)

    # Datasets
    cs.store(name="base_dataset_config_name", group="dataset", node=BaseDatasetConfig,)

    # Optimizers
    cs.store(name="base_optim_config_name", group="optim_f_theta", node=BaseOptimConfig)
    cs.store(name="adamw_name", group="optim_f_theta", node=AdamwConfig)
    cs.store(name="base_optim_config_name", group="optim_log_bias", node=BaseOptimConfig)
    cs.store(name="adamw_name", group="optim_log_bias", node=AdamwConfig)
    cs.store(name="base_optim_config_name", group="optim_proposal", node=BaseOptimConfig)
    cs.store(name="adamw_name", group="optim_proposal", node=AdamwConfig)
    cs.store(name="base_optim_config_name", group="optim_base_dist", node=BaseOptimConfig)
    cs.store(name="adamw_name", group="optim_base_dist", node=AdamwConfig)

    # Scheduler
    cs.store(name="base_scheduler_config_name", group="scheduler_f_theta", node=BaseSchedulerConfig,)
    cs.store(name="base_scheduler_config_name", group="scheduler_log_bias", node=BaseSchedulerConfig,)
    cs.store(name="base_scheduler_config_name", group="scheduler_proposal", node=BaseSchedulerConfig,)
    cs.store(name="base_scheduler_config_name", group="scheduler_base_dist", node=BaseSchedulerConfig,)

    # Proposal training
    cs.store(name="base_proposal_training_config_name", group="proposal_training", node=BaseProposalTrainingConfig,)

    # Base Proposal
    cs.store(name="base_proposal_config_name", group="proposal", node=BaseProposalConfig)

    # Base distributions
    cs.store(name="base_distribution_config_name",group="base_distribution",node=BaseBaseDistributionConfig,)

    # Base Default Proposal
    cs.store(name="base_default_proposal_config_name", group="default_proposal", node=BaseProposalConfig,)

    # Energy
    cs.store(name="base_energy_config_name", group="energy", node=BaseEnergyConfig,)

    # Explicit bias
    cs.store(name="base_explicit_bias_config_name", group="explicit_bias", node=BaseExplicitBiasConfig,)

    # Feature extractor
    cs.store(name="base_feature_extractor_config_name", group="feature_extractor", node=BaseFeatureExtractorConfig,)

    # Trainer
    cs.store(name="base_train_config_name", group="train", node=BaseTrainConfig)

    # Samplers
    cs.store(name="base_sampler_config_name", group="sampler", node=BaseSamplerConfig)
    cs.store(name="nuts_name", group="sampler", node=NutsConfig)

    # Machine
    cs.store(name="karolina_name", group="machine", node=Machine)
    cs.store(name="local_name", group="machine", node=Machine)
    cs.store(name="dtu_cluster_name", group="machine", node=Machine)


@hydra.main(version_base="1.1", config_name="config", config_path="conf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    store_main()
    main()
