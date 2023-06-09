from .DistributionEstimation import SelfNormalizedTrainer, NCETrainer, ProposalTrainer, ScoreMatchingTrainer, DenoisingScoreMatchingTrainer
from .Regression import RegressionTrainerSelfNormalized, ProposalRegressionTrainer, RegressionTrainerNCE

dic_trainer = {
    'self_normalized': SelfNormalizedTrainer,
    'just_proposal' : ProposalTrainer,
    'nce' : NCETrainer,
    'score_matching' : ScoreMatchingTrainer,
    'denoising_score_matching' : DenoisingScoreMatchingTrainer,
    
}

dic_trainer_regression = {
    'self_normalized': RegressionTrainerSelfNormalized,
    'just_proposal' : ProposalRegressionTrainer,
    'nce' : RegressionTrainerNCE
}