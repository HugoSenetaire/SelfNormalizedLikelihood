from .DistributionEstimation import SelfNormalized, ELBO

dic_ebm = {
    'self_normalized': SelfNormalized,
    'elbo' : ELBO
}