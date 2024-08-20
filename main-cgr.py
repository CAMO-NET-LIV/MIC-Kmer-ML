import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.cnn2 import CNN2
from src.dataset.file_label import FileLabel
from src.dataset.loader import Loader
from models.cnn import CNN
import pandas as pd
import numpy as np
import argparse
import ray
import math

parser = argparse.ArgumentParser()
parser.add_argument('--label-file', type=str, required=True)
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--antibiotic', type=str, required=True)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--fold', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--workers', type=int, default=38)
args = parser.parse_args()

file_label = FileLabel(
    '../volatile/cgr_labels/cgr_label.csv',
    '../volatile/cgr/',
    antibiotic=args.antibiotic
)

loader = Loader(file_label, n_fold=10)

result = pd.DataFrame(columns=['genome_id', 'true', 'pred'])
fold = 1
for train_kmer, test_kmer, train_label, test_label, train_genome_id, test_genome_id in loader.get_kmer_dataset(10):
    print(f'Fold {fold}')
    train_dataset = TensorDataset(torch.tensor(train_kmer, dtype=torch.float32),
                                  torch.tensor(train_label, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_kmer, dtype=torch.float32),
                                 torch.tensor(test_label, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_dim = train_kmer.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN2(input_dim, device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        rmse_loss = math.sqrt(running_loss / len(train_loader))  # Root Mean Square Error (RMSE)

        # Evaluation on test data during training
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_loss += criterion(outputs.squeeze(), batch_y).item()

        rmse_test_loss = math.sqrt(test_loss / len(test_loader))  # Root Mean Square Error (RMSE)

        # Combine training and test loss into a single line
        print(f'Epoch [{epoch + 1}/{args.epochs}], Train Loss: {rmse_loss:.4f}, Test Loss: {rmse_test_loss:.4f}')

    # Evaluation
    model.eval()
    test_loss = 0.0
    test_preds = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            test_loss += loss.item()
            test_preds.append(outputs.cpu().squeeze().numpy())

    rmse_test_loss = math.sqrt(test_loss / len(test_loader))  # Root Mean Square Error (RMSE)
    test_preds = np.concatenate(test_preds)
    print(f'Test Loss: {rmse_test_loss:.4f}')  # Printing RMSE loss instead of MSE

    # Add results to the dataframe
    for genome_id, true_label, pred_label in zip(test_genome_id, test_label, test_preds):
        result = pd.concat([result, pd.DataFrame({
            'genome_id': [genome_id],
            'true': [true_label],
            'pred': [pred_label]
        })], ignore_index=True)

    fold += 1

result.to_csv(f'results_{args.antibiotic}.csv', index=False)
