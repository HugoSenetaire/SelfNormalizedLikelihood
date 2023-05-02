from .DistributionEstimation.trainer_self_normalized import LitSelfNormalized
from .Regression import RegressionTrainerSelfNormalized

dic_trainer = {
    'self_normalized': LitSelfNormalized
}

dic_trainer_regression = {
    'self_normalized': RegressionTrainerSelfNormalized
}