import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, input_dim, device):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.last_channel = 256

        self.reducer = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.Conv1d(256, self.last_channel, kernel_size=16, stride=16),
            nn.ReLU()
        )

        arch = [16, 16, 16, 16]

        self.regressor = nn.Sequential(
            nn.Linear(self.calc_input_feature_size(arch), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = self.reducer(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

    def calc_input_feature_size(self, arch):
        output_size = self.input_dim
        for kernel_size, stride in zip(arch, arch):
            output_size = ((output_size - kernel_size) // stride) + 1
        num = output_size * self.last_channel  # multiply by the number of output channels of the last conv layer
        print(f'Regressor layer: {num}')

        return num
