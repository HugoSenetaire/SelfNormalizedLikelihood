from .DistributionEstimation import SelfNormalizedTrainer, NCETrainer, ProposalTrainer
from .Regression import RegressionTrainerSelfNormalized, ProposalRegressionTrainer, RegressionTrainerNCE

dic_trainer = {
    'self_normalized': SelfNormalizedTrainer,
    'just_proposal' : ProposalTrainer,
    'nce' : NCETrainer,
    
}

dic_trainer_regression = {
    'self_normalized': RegressionTrainerSelfNormalized,
    'just_proposal' : ProposalRegressionTrainer,
    'nce' : RegressionTrainerNCE
}