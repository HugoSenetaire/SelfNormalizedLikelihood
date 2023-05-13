def log_prob_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape
    return - (log_prob_proposal).mean()

def kl_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape
    return log_estimate_z.mean()

def log_prob_kl_loss_regression(log_prob_proposal, log_estimate_z):
    assert log_prob_proposal.shape == log_estimate_z.shape

    return - 0.5*(log_prob_proposal - log_estimate_z).mean()

def log_prob_loss(log_prob_proposal, log_estimate_z):
    # print(log_prob_proposal.shape)
    # print(log_estimate_z.shape)

    return - (log_prob_proposal).mean()

def kl_loss(log_prob_proposal, log_estimate_z):

    return log_estimate_z.mean()

def log_prob_kl_loss(log_prob_proposal, log_estimate_z):
    return - 0.5*(log_prob_proposal - log_estimate_z).mean()