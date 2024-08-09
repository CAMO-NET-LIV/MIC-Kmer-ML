import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DNP(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout_rate=0.3):
        super(DNP, self).__init__()
        self.dropout_rate = dropout_rate
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize the weights of the first layer to 0
        nn.init.constant_(self.net[0].weight, 0)

    def forward(self, x, apply_dropout=False):
        for layer in self.net:
            x = layer(x)
            if apply_dropout and isinstance(layer, nn.ReLU):
                x = self.dropout(x)
        return x
