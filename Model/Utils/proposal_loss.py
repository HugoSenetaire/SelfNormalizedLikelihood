def log_prob_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape
    return - (log_prob_proposal).mean()

def kl_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape
    return log_estimate_z.mean()

def log_prob_kl_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape

    return - 0.5*(log_prob_proposal - log_estimate_z).mean()


def proposal_loss_regression_getter(name):
    if name is None :
        return None
    elif name == 'log_prob':
        return log_prob_loss_regression
    elif name == 'kl':
        return kl_loss_regression
    elif name == 'log_prob_kl':
        return log_prob_kl_loss_regression
    else:
        raise NotImplementedError

def log_prob_loss(log_prob_proposal, log_estimate_z):
    return - (log_prob_proposal).mean()

def kl_loss(log_prob_proposal, log_estimate_z):
    return log_estimate_z.mean()

def log_prob_kl_loss(log_prob_proposal, log_estimate_z):
    return - 0.5*(log_prob_proposal - log_estimate_z).mean()


def proposal_loss_getter(name):
    if name is None :
        return None
    elif name == 'log_prob':
        return log_prob_loss
    elif name == 'kl':
        return kl_loss
    elif name == 'log_prob_kl':
        return log_prob_kl_loss
    else:
        raise NotImplementedError