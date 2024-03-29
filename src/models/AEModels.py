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
            encoder.append(nn.Tanh())
            input_dim = output_dim
        self.encoder = nn.ModuleList(encoder)

        self.latent = nn.Linear(input_dim, latent_dim, dtype=torch.float32) 

        # decoder (x ~ gausian)
        decoder, input_dim = [], latent_dim
        for i, output_dim in enumerate(self.hidden_layers+[self.num_items]):
            decoder.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.decoder = nn.ModuleList(decoder)

        # initialize parameters
        # self._init_params()

    def _init_params(self):
        for child in self.children():
            if isinstance(child, nn.Linear):
                #nn.init.kaiming_uniform_(child.weight)
                nn.init.xavier_uniform_(child.weight)
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
        self.dropout_ratio = dropout
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def add_noise(self, x):
        return x + self.noise_factor * x.new_tensor(torch.randn_like(x))

    def forward(self, x):
        if self.training:
            x = self.add_noise(x)
            x = self.dropout(x)
        latent = self.encode(x)
        output = self.decode(latent)
        return output


class VAE(AE):

    def __init__(self, num_items, latent_dim=200, hidden_layers=[600,], noise_factor=0.5, dropout=.5, denosing=False):
        super().__init__(num_items, latent_dim, hidden_layers, dropout)
        self.code_mean = nn.Linear(self.hidden_layers[-1], self.latent_dim)
        self.code_logvar = nn.Linear(self.hidden_layers[-1], self.latent_dim)
        self.denosing = denosing
        if self.denosing:
            self.noise_factor = noise_factor
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def add_noise(self, x):
        return x + self.noise_factor * x.new_tensor(torch.randn_like(x))

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.code_mean(x), self.code_logvar(x)

    def reparameterize(self, mean, logvar):
        stddev = torch.exp(.5 * logvar)
        if self.training:
            eps = torch.randn_like(stddev)
        else:
            eps = 0
        z = mean + stddev * eps
        return z

    def forward(self, x):
        if self.denosing and self.training:
            x = self.add_noise(x)
            x = self.dropout(x)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z)
        return output, mean, logvar
