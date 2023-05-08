def log_prob_loss_regression(log_prob_proposal, log_estimate_z):
    return - (log_prob_proposal).mean()

def kl_loss_regression(log_prob_proposal, log_estimate_z):
    return - (log_estimate_z).mean()

def log_prob_kl_loss_regression(log_prob_proposal, log_estimate_z):
    return - (log_prob_proposal + log_estimate_z).mean()

def log_prob_loss(log_prob_proposal, log_estimate_z):
    return - (log_prob_proposal).mean()

def kl_loss(log_prob_proposal, log_estimate_z):
    return - (log_estimate_z).mean()

def log_prob_kl_loss(log_prob_proposal, log_estimate_z):
    return - (log_prob_proposal + log_estimate_z).mean()