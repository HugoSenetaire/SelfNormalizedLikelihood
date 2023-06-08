from .DistributionEstimation import SelfNormalizedTrainer, NCETrainer, ProposalTrainer, ScoreMatchingTrainer, DenoisingScoreMatchingTrainer
from .Regression import RegressionTrainerSelfNormalized, ProposalRegressionTrainer, RegressionTrainerNCE

from .DistributionEstimationDualProposal import DualDenoisingScoreMatchingTrainer, DualNCETrainer, DualProposalTrainer, DualScoreMatchingTrainer,DualSelfNormalizedTrainer

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

dic_trainer_dual = {
    'self_normalized': DualSelfNormalizedTrainer,
    'just_proposal' : DualProposalTrainer,
    'nce' : DualNCETrainer,
    'score_matching' : DualScoreMatchingTrainer,
    'denoising_score_matching' : DualDenoisingScoreMatchingTrainer,
}

