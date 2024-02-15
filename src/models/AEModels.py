import numpy as np

import torch
import torch.nn as nn


class AE(nn.Module):

    def __init__(self, num_items, latent_dim=200, hidden_layers=[600,]):

        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        
        # encoder
        encoder, input_dim = [], num_items
        for output_dim in self.hidden_layers:
            encoder.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.encoder = nn.ModuleList(encoder)

        self.latent = nn.Linear(input_dim, latent_dim, dtype=torch.float32) 

        # decoder (x ~ gausian)
        decoder, input_dim = [], latent_dim
        for output_dim in self.hidden_layers+[self.num_items]:
            decoder.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.decoder = nn.ModuleList(decoder)

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.latent(x)

    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output
