from .importance_weighted_ebm import ImportanceWeightedEBM
from .ais_importance_weighted_ebm import AISImportanceWeightedEBM
dic_ebm = {
    "standard": ImportanceWeightedEBM,
    "ais": AISImportanceWeightedEBM,
}
