import torch.nn as nn
import torch


class CNN2(nn.Module):
    def __init__(self, input_dim, batch_size, device):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.device = device
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=16)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=16, stride=16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=16, stride=16)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 1)

    def calc_output_size(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1

    def determine_conv2_repeats(self):
        output_size = self.calc_output_size(self.input_dim, 16, 16)
        conv2_repeats = 0
        while True:
            output_size = self.calc_output_size(output_size, 16, 16)
            conv2_repeats += 1
            if 4096 <= output_size * 32 <= 8192:
                break

        # -1 because the last conv3 layer is not counted
        return conv2_repeats - 1

    def forward(self, x):
        conv2_repeats = self.determine_conv2_repeats()
        x = x.view(-1, 1, self.input_dim)
        x = torch.relu(self.conv1(x))
        for _ in range(conv2_repeats):
            x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = torch.relu(
            self.get_first_fully_connected_layer([16, 16] + [16] * conv2_repeats)(x)
        )
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def calc_input_feature_size(self, arch):
        output_size = self.input_dim
        for kernel_size, stride in zip(arch, arch):
            output_size = ((output_size - kernel_size) // stride) + 1
        return output_size * 32  # multiply by the number of output channels of the last conv layer

    def get_first_fully_connected_layer(self, arch):
        return nn.Linear(self.calc_input_feature_size(arch), 2048).to(self.device)
