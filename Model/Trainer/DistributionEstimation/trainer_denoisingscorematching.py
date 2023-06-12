import torch
from .abstract_trainer import AbstractDistributionEstimation
import torch.autograd

class DenoisingScoreMatchingTrainer(AbstractDistributionEstimation):
    """
    Trainer for an EBM using denoising score matching.
    """
    def __init__(self, ebm, args_dict, complete_dataset = None, nb_sample_train_estimate=1024, **kwargs):
        super().__init__(ebm = ebm,
                        args_dict = args_dict,
                        complete_dataset = complete_dataset,
                        nb_sample_train_estimate=nb_sample_train_estimate,
                        **kwargs)
        if 'sigma' in self.args_dict.keys():
            self.sigma = self.args_dict['sigma']
        else :
            self.sigma = .1

    def denoising_score_matching(self, data,):
        data = data.flatten(1)
        data.requires_grad_(True)
        vector = torch.randn_like(data) * self.sigma
        perturbed_data = data + vector
        energy_perturbed_data, dic = self.ebm.calculate_energy(perturbed_data)
        logp = -energy_perturbed_data  # logp(x)

        dlogp = self.sigma ** 2 * torch.autograd.grad(logp.sum(), perturbed_data, create_graph=True)[0]
        kernel = vector
        loss = torch.norm(dlogp + kernel, dim=-1) ** 2
        loss = loss.mean() / 2.
       
        return loss, dic


    def training_step(self, batch, batch_idx):
        ebm_opt, proposal_opt = self.optimizers_perso()
        x = batch['data']
        if hasattr(self.ebm.proposal, 'set_x'):
            self.ebm.proposal.set_x(x)


        loss_total, dic_output = self.denoising_score_matching(x)
        self.log("train_loss", loss_total.mean())
        ebm_opt.zero_grad()
        self.manual_backward(loss_total.mean(), retain_graph=False, )
        ebm_opt.step()

        ebm_opt.zero_grad()

        estimate_log_z, dic = self.ebm.estimate_log_z(x, self.num_samples_train)
        estimate_log_z = estimate_log_z.mean()
        dic_output.update(dic)


        # Update the parameters of the proposal
        self._proposal_step(x = x, estimate_log_z = estimate_log_z, proposal_opt = proposal_opt, dic_output=dic_output,)
        self.post_train_step_handler(x, dic_output,)

        
        return loss_total
    

