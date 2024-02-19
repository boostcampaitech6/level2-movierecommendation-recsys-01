# loss.py
# pointwise Loss
# pairwise Loss

import torch
import torch.nn as nn


class MultiAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        # 재구성 손실
        recon_loss = -torch.mean(torch.sum(nn.functional.log_softmax(pred, -1) * true, -1))
        return recon_loss 


class VAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.kl_anneal = args.kl_anneal
        if self.kl_anneal:
            self.anneal_beta_max = args.anneal_beta_max
            self.anneal_total_steps = args.anneal_total_steps # batch_size * epochs
            self.beta = 0
            self.update_count = 0


    def forward(self, pred, true, mean, logvar, train=False):
        # 재구성 손실
        recon_loss = self.loss(pred, true)
        # KL 발산: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_distance = -0.5 * torch.mean(
            torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

        if self.kl_anneal:
            total_loss = recon_loss + self.beta * kl_distance
            self.set_beta(train)
        else:
            total_loss = recon_loss + kl_distance

        return total_loss
