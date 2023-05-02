from .DistributionEstimation import SelfNormalized, ELBO
from .Regression import SelfNormalizedRegression, EUBORegression

dic_ebm = {
    'self_normalized': SelfNormalized,
    'elbo' : ELBO
}

dic_ebm_regression = {
    'self_normalized': SelfNormalizedRegression,
    'eubo' : EUBORegression,
}