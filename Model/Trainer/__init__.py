from .DistributionEstimation.trainer_self_normalized import LitSelfNormalized
from .Regression import RegressionTrainerSelfNormalized, ProposalRegressionTrainer, RegressionTrainerNCE

dic_trainer = {
    'self_normalized': LitSelfNormalized
}

dic_trainer_regression = {
    'self_normalized': RegressionTrainerSelfNormalized,
    'just_proposal' : ProposalRegressionTrainer,
    'nce' : RegressionTrainerNCE
}