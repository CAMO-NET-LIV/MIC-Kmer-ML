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
from models.cnn2 import CNN2
from models.mlp import MLP
from nn_data_loader import CustomDataset
from util import is_essential_agreement
from torchsummary import summary

from utils.eval import evaluate_model, model_inference
from utils.model_saver import ModelSaver

# Argument Parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--kmer', type=int, default=12, help='k-mer value')
parser.add_argument('--model', type=str, default='cnn', help='Model to use')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
args = parser.parse_args()

# Check if CUDA is available, otherwise use CPU
device = torch.device(args.device)

# Configuration constants
LABEL_FILE = 'labels-cleaned.csv'
DATA_DIR = f'../volatile/genome-data-ignore/processed_{args.kmer}mer_count/'
RESULT_DIR = '../volatile/results/'
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.0004

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=12)

# Initialize the model, loss function, and optimizer
model_name = args.model.lower()
if model_name == 'cnn':
    model = CNN(input_dim=4 ** args.kmer, batch_size=BATCH_SIZE, device=device)
if model_name == 'cnn2':
    model = CNN2(input_dim=4 ** args.kmer, batch_size=BATCH_SIZE, device=device)
elif model_name == 'MLP':
    model = MLP(input_dim=4 ** args.kmer)
else:
    raise Exception('Invalid model type. Choose between "CNN" and "MLP".')

saver = ModelSaver(model, RESULT_DIR, model_name, args.kmer, args.device, BATCH_SIZE, EPOCHS)
try:
    current_epoch = saver.load_weight()
except FileNotFoundError:
    current_epoch = 0
    print('No saved model found. Training from scratch.')

model = model.to(device)  # Move the model to GPU
print('running on:', device)

# Check if multiple GPUs are available and wrap the model with DataParallel
# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs")
#     model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(current_epoch, EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # save model weights every epoch
    saver.save_weight(epoch)

    # Evaluation on training data
    train_rmse, train_ea = evaluate_model(model, train_loader, device)

    # Evaluation on testing data
    test_rmse, test_ea = evaluate_model(model, test_loader, device)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], '
          f'Train RMSE: {train_rmse:.4f}, Train EA: {train_ea:.4f}, '
          f'Test RMSE: {test_rmse:.4f}, Test EA: {test_ea:.4f}')

# Final Evaluation
final_rmse, final_ea = evaluate_model(model, test_loader, device)
print(f'Final RMSE: {final_rmse:.4f}, Final EA: {final_ea:.4f}')

predictions, actuals = model_inference(model, test_loader, device)
ea = is_essential_agreement(actuals, predictions)

dataframe = {
    'actual': actuals,
    'predicted': predictions,
    'ea': ea
}

df = pd.DataFrame(dataframe)
df.to_csv(os.path.join(RESULT_DIR, saver.get_file_name() + '.csv'), index=False)
