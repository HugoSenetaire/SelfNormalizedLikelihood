from .DistributionEstimation import (
    KALE,
    VERA,
    AnnealedImportanceSampling,
    DenoisingScoreMatchingTrainer,
    NCETrainer,
    PersistentReplayLangevin,
    ProposalTrainer,
    ScoreMatchingTrainer,
    SelfNormalizedTrainer,
    ShortTermLangevin,
    LogTrainer,
    LogTrainerSelfNormalized,
    SelfNormalizedShortTermTrainer,
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
    "ais_self_normalized": AnnealedImportanceSampling,
    "log_trainer": LogTrainer,
    "log_trainer_self_normalized": LogTrainerSelfNormalized,
    "self_normalized_short_term": SelfNormalizedShortTermTrainer,
}
