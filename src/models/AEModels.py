import numpy as np

import torch
import torch.nn as nn


class AE(nn.Module):

    def __init__(self, num_items, latent_dim=200, hidden_layers=[600,], dropout=.5):

        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.dropout_ratio = dropout
        
        # encoder
        encoder, input_dim = [], num_items
        for output_dim in self.hidden_layers:
            encoder.append(nn.Linear(input_dim, output_dim))
            #encoder.append(nn.Sigmoid()) #nn.ReLU()
            input_dim = output_dim
        self.encoder = nn.ModuleList(encoder)

        self.latent = nn.Linear(input_dim, latent_dim, dtype=torch.float32) 

        # decoder (x ~ gausian)
        decoder, input_dim = [], latent_dim
        for output_dim in self.hidden_layers+[self.num_items]:
            decoder.append(nn.Linear(input_dim, output_dim))
            encoder.append(nn.Dropout(p=self.dropout_ratio))
            #decoder.append(nn.Sigmoid()) #nn.ReLU()
            input_dim = output_dim
        #decoder.pop()
        self.decoder = nn.ModuleList(decoder)

        # initialize parameters
        # self._init_params()

    def _init_params(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.kaiming_uniform_(child.weight)
                nn.init.uniform_(child.bias)

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


class DAE(AE):

    def __init__(self, num_items, latent_dim=200, hidden_layers=[600,], noise_factor=0.5, dropout=.5):
        super().__init__(num_items, latent_dim, hidden_layers, dropout)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        return x + self.noise_factor * x.new_tensor(torch.randn_like(x))

    def forward(self, x):
        if self.training:
            x = self.add_noise(x)
        latent = self.encode(x)
        output = self.decode(latent)
        return output
