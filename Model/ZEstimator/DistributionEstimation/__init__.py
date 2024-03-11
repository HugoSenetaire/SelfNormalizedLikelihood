from .importance_weighted_ebm import ImportanceZEstimator
from .ais_importance_weighted_ebm import AISZEstimator
dic_z_estimator = {
    "standard": ImportanceZEstimator,
    "ais": AISZEstimator,
}
