import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import argparse

from dataset_manager import DatasetManager
from models.cnn import CNN
from models.mlp import MLP
from nn_data_loader import CustomDataset
from util import is_essential_agreement
from torchsummary import summary

# Argument Parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--kmer', type=int, default=12, help='k-mer value')
parser.add_argument('--model', type=str, default='CNN', help='Model to use')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
args = parser.parse_args()

# Check if CUDA is available, otherwise use CPU
device = torch.device(args.device)

# Configuration constants
LABEL_FILE = 'labels-cleaned.csv'
DATA_DIR = f'../volatile/genome-data-ignore/processed_{args.kmer}mer_count/'
RESULT_DIR = 'result/'
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.0005


def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().squeeze().numpy())
            actuals.extend(batch_y.cpu().squeeze().numpy())

        mse_loss = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse_loss)
        ea = np.mean(is_essential_agreement(actuals, predictions))
    return rmse, ea


dataset_manager = DatasetManager(LABEL_FILE, DATA_DIR)
train_files, test_files, train_labels, test_labels = dataset_manager.prepare_train_test_path()

train_dataset = CustomDataset(
    file_names=train_files,
    labels=train_labels,
    data_dir=DATA_DIR,
)

test_dataset = CustomDataset(
    file_names=test_files,
    labels=test_labels,
    data_dir=DATA_DIR,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16)

# Initialize the model, loss function, and optimizer
model_name = args.model.lower()
if model_name == 'cnn':
    model = CNN(input_dim=4 ** args.kmer, batch_size=BATCH_SIZE, device=device)
elif model_name == 'MLP':
    model = MLP(input_dim=4 ** args.kmer)
else:
    raise Exception('Invalid model type. Choose between "CNN" and "MLP".')

model = model.to(device)  # Move the model to GPU
print('running on:', device)

# Check if multiple GPUs are available and wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Evaluation on training data
    train_rmse, train_ea = evaluate_model(model, train_loader)

    # Evaluation on testing data
    test_rmse, test_ea = evaluate_model(model, test_loader)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], '
          f'Train RMSE: {train_rmse:.4f}, Train EA: {train_ea:.4f}, '
          f'Test RMSE: {test_rmse:.4f}, Test EA: {test_ea:.4f}')

# Final Evaluation
final_rmse, final_ea = evaluate_model(model, test_loader)
print(f'Final RMSE: {final_rmse:.4f}, Final EA: {final_ea:.4f}')
