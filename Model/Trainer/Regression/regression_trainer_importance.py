from .abstract_regression_trainer import AbstractRegression

class RegressionTrainerSelfNormalized(AbstractRegression):
    def __init__(self, ebm, cfg, complete_dataset = None, **kwargs):
        super().__init__(ebm, cfg, complete_dataset = complete_dataset, **kwargs)


    def training_step(self, batch, batch_idx,):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()

        x = batch['data']
        y = batch['target']
        x_feature = self.ebm.feature_extractor(x)
        energy_data, dic_output = self.ebm.calculate_energy(x, y, x_feature=x_feature)
        energy_data.reshape(x.shape[0], 1, -1)
        log_prob_proposal_data = self.ebm.log_prob_proposal(x,y, x_feature=x_feature).reshape(x.shape[0], 1, -1)
        estimate_log_z, dic=self.ebm.estimate_log_z(x, self.num_samples_train, x_feature=x_feature)
        dic_output.update(dic)
        dic_output.update({'log_prob_proposal_data': log_prob_proposal_data.mean()})
        estimate_log_z = estimate_log_z.reshape(x.shape[0], 1, -1)
        loss_total = (energy_data + estimate_log_z).mean()

        # Update the parameters
        ebm_opt.zero_grad()
        proposal_opt.zero_grad()
        self.manual_backward(loss_total, retain_graph=True, )
        self.proposal_step(log_prob_proposal_data = log_prob_proposal_data,
                        estimate_log_z = estimate_log_z,
                        proposal_opt = proposal_opt, 
                        dic_output= dic_output,)
        ebm_opt.step()
        dic_output.update(dic)

        self.log('train_loss', loss_total)
        for key in dic_output.keys():
            self.log(f'train_{key}', dic_output[key].mean())
        return loss_total



    
    
        

