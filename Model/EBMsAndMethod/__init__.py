from .self_normalized import *
from .elbo import ELBO

dic_ebm = {
    'self_normalized': SelfNormalized,
    'elbo' : ELBO
}