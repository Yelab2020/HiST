import torch
import torch.nn as nn
import torch.nn.functional as F



class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, censors):
        
        riskmax = F.normalize(risk, p=2, dim=0)

        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))

        return loss