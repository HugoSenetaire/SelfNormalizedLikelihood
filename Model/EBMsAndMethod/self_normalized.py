import torch.nn as nn
import torch
from .importance_weighted_ebm import ImportanceWeightedEBM

class SelfNormalized(ImportanceWeightedEBM):
    def __init__(self, energy, proposal, num_sample_proposal, base_dist=None, switch_mode_index = None, **kwargs):
        super(SelfNormalized, self).__init__(energy=energy, proposal=proposal, num_sample_proposal=num_sample_proposal, base_dist=base_dist, **kwargs)
        if switch_mode_index is None or switch_mode_index<=0 :
            self.type_z = "exp"
        else :
            self.type_z = "log"
        self.switch_mode_index = switch_mode_index
        
    


    def switch_mode(self, ):
        """
        When changing mode, we need to re-estimate the partition function.
        """
        super(SelfNormalized, self).switch_mode()
        self.type_z = "exp"




    
