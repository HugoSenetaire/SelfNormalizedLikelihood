from .DistributionEstimation import (
    VERA,
    DenoisingScoreMatchingTrainer,
    NCETrainer,
    ProposalTrainer,
    ScoreMatchingTrainer,
    SelfNormalizedTrainer,
)
from .Regression import (
    ProposalRegressionTrainer,
    RegressionTrainerNCE,
    RegressionTrainerSelfNormalized,
)

dic_trainer = {
    "self_normalized": SelfNormalizedTrainer,
    "just_proposal": ProposalTrainer,
    "nce": NCETrainer,
    "score_matching": ScoreMatchingTrainer,
    "denoising_score_matching": DenoisingScoreMatchingTrainer,
    "vera": VERA,
}

dic_trainer_regression = {
    "self_normalized": RegressionTrainerSelfNormalized,
    "just_proposal": ProposalRegressionTrainer,
    "nce": RegressionTrainerNCE,
}
