import torch.nn as nn
import torch


class CNN2(nn.Module):
    def __init__(self, input_dim, device):
        super(CNN2, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.last_channel = 32

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=32, stride=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),  # Add batch normalization layer
            nn.Conv1d(32, 32, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.BatchNorm1d(32),  # Add batch normalization layer
            nn.Conv1d(32, 32, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.Conv1d(32, self.last_channel, kernel_size=16, stride=16),
            nn.ReLU(),
        )

        arch = [32, 16, 16, 16]

        self.classifier = nn.Sequential(
            nn.Linear(self.calc_input_feature_size(arch), 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout layer
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def calc_input_feature_size(self, arch):
        output_size = self.input_dim
        for kernel_size, stride in zip(arch, arch):
            output_size = ((output_size - kernel_size) // stride) + 1
        num = output_size * self.last_channel  # multiply by the number of output channels of the last conv layer
        print(f'Regressor layer: {num}')

        return num
