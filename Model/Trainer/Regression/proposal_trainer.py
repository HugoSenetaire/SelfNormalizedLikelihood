from .abstract_regression_trainer import AbstractRegression

class ProposalRegressionTrainer(AbstractRegression):
    def __init__(self, ebm, cfg, complete_dataset = None, **kwargs):
        '''
        Train a only the proposal from the model
        '''
        cfg.proposal_training.train_proposal = True
        super().__init__(ebm, cfg, complete_dataset = complete_dataset, **kwargs)

    def training_step(self, batch, batch_idx,):
        # Get parameters
        ebm_opt, proposal_opt = self.optimizers()

        x = batch['data']
        y = batch['target']
        x_feature = self.ebm.feature_extractor(x)
        log_prob_proposal = self.ebm.log_prob_proposal(x,y, x_feature=x_feature)
        proposal_opt.zero_grad()
        proposal_loss = self.proposal_loss(log_prob_proposal=log_prob_proposal, log_estimate_z=None)
        self.manual_backward(proposal_loss, inputs= list(self.ebm.proposal.parameters()))
        proposal_opt.step()

        self.log('train_loss', proposal_loss)
        return proposal_loss


        
        
    
    
        

