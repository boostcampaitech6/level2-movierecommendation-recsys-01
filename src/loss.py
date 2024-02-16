# loss.py
# pointwise Loss
# pairwise Loss

import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, true, mean, logvar):
        recon_loss = self.loss(pred, true)
        # KL 발산: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_distance = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_distance 
