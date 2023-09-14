from .DistributionEstimation import (
    KALE,
    VERA,
    DenoisingScoreMatchingTrainer,
    NCETrainer,
    ProposalTrainer,
    ScoreMatchingTrainer,
    SelfNormalizedTrainer,
    ShortTermLangevin,
    PersistentReplayLangevin,
)
from .Regression import (
    ProposalRegressionTrainer,
    RegressionTrainerNCE,
    RegressionTrainerSelfNormalized,
)

dic_trainer = {
    "self_normalized": SelfNormalizedTrainer,
    "short_term_langevin": ShortTermLangevin,
    "persistent_replay_langevin": PersistentReplayLangevin,
    "just_proposal": ProposalTrainer,
    "nce": NCETrainer,
    "score_matching": ScoreMatchingTrainer,
    "denoising_score_matching": DenoisingScoreMatchingTrainer,
    "vera": VERA,
    "kale": KALE,
}

dic_trainer_regression = {
    "self_normalized": RegressionTrainerSelfNormalized,
    "just_proposal": ProposalRegressionTrainer,
    "nce": RegressionTrainerNCE,
}
