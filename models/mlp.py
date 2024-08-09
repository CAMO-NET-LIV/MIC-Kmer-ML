import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc9(x))
        x = self.fc10(x)
        x = x.view(-1, 1)
        return x
