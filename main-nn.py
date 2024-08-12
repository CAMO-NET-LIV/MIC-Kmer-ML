import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from dataset_manager import DatasetManager
from util import is_essential_agreement
from utils.args import parse_arguments
from utils.data_distribution import DataDistributionManager
from utils.eval import evaluate_model, model_inference
from utils.model_configs import get_model, get_model_with_dim
from utils.model_saver import ModelSaver
from utils.norm import lq_norm, gradients_batch_norm


def load_model_weights(saver):
    try:
        current_epoch = saver.load_weight()
        print(f'Model loaded from epoch {current_epoch}')
    except FileNotFoundError:
        current_epoch = 0
        print('No saved model found. Training from scratch.')
    except Exception as e:
        print(f'Error loading model: {e}')
        current_epoch = 0
    return current_epoch


def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, saver, device, epochs, rank,
                             world_size):
    for epoch in range(current_epoch, epochs):
        model.train()
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        if rank == 0:
            train_rmse, train_ea = evaluate_model(model, train_loader, device, scale=SCALE)
            test_rmse, test_ea = evaluate_model(model, test_loader, device, scale=SCALE)
            print(f'Epoch [{epoch + 1}/{epochs}], Train RMSE: {train_rmse:.8f}, Train EA: {train_ea:.8f}, '
                  f'Test RMSE: {test_rmse:.8f}, Test EA: {test_ea:.8f}')
            saver.save_weight(epoch)


def train_and_evaluate_dnp(model, train_loader, test_loader, input_dim, criterion, optimizer, device, epochs, rank,
                           world_size, patience=5):
    max_features = 128
    selected_features = []
    all_features = list(range(input_dim))

    # calc input sum and count

    x_sum = torch.zeros(input_dim).to(device)
    count = 0
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        x_sum += torch.sum(batch_x, dim=0)
        count += len(batch_x)

    # calc avg
    x_avg = x_sum / count

    for _ in range(max_features):
        model.train()

        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = model(batch_x, apply_dropout=False)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Manually zero out gradients for non-selected features
                for name, param in model.named_parameters():
                    if name == 'net.0.weight':
                        mask = torch.tensor([i not in selected_features for i in range(param.size(1))],
                                            device=device)
                        param.grad[:, mask] = 0

                optimizer.step()

            if rank == 0:
                train_rmse, train_ea = evaluate_model(model, train_loader, device, scale=SCALE)
                test_rmse, test_ea = evaluate_model(model, test_loader, device, scale=SCALE)
                print(f'Epoch [{epoch + 1}/{epochs}], Train RMSE: {train_rmse:.8f}, Train EA: {train_ea:.8f}, '
                      f'Test RMSE: {test_rmse:.8f}, Test EA: {test_ea:.8f}')

                # Early stopping logic
                if test_rmse < best_loss:
                    best_loss = test_rmse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        print('Calculating gradients for candidates')
        # Bp multiple times and calculate total (avg) gradients
        gradients = torch.zeros(input_dim).to(device)
        for _ in range(5):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = model(batch_x, apply_dropout=True)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()

                # Accumulate gradients for non-selected features
                for name, param in model.named_parameters():
                    if name == 'net.0.weight':
                        mask = torch.tensor([i not in selected_features for i in range(param.size(1))],
                                            device=device)
                        gradients[mask] += lq_norm(param.grad[:, mask])

        # Select feature with maximum gradient norm
        # gradients = gradients / x_avg  # Normalize by sum of feature values
        candidate_features = [feature for feature in all_features if feature not in selected_features]
        best_feature = candidate_features[torch.argmax(gradients[candidate_features])]
        selected_features.append(best_feature)

        print(f'Selected feature {best_feature}')
        print(f'Selected features: {selected_features}')

        # Xavier initialization for newly selected feature
        for name, param in model.named_parameters():
            if name == 'net.0.weight':
                nn.init.xavier_uniform_(param[:, best_feature:best_feature + 1])

    return selected_features


# Rest of the code remains the same

def final_evaluation(model, test_loader, device, saver, result_dir):
    final_rmse, final_ea = evaluate_model(model, test_loader, device, scale=SCALE)
    print(f'Final RMSE: {final_rmse:.8f}, Final EA: {final_ea:.8f}')

    predictions, actuals = model_inference(model, test_loader, device)
    ea = is_essential_agreement(actuals, predictions)

    dataframe = {'actual': actuals, 'predicted': predictions, 'ea': ea}
    df = pd.DataFrame(dataframe)
    df.to_csv(os.path.join(result_dir, saver.get_file_name() + '.csv'), index=False)


if __name__ == "__main__":
    args = parse_arguments()

    SCALE = False
    LABEL_FILE = 'labels-cleaned.csv'
    DATA_DIR = f'../volatile/genome-data-ignore/processed_{args.km}mer_count/'
    RESULT_DIR = '../volatile/results/'
    leaning_rate = args.lr * args.world_size

    print('Arguments:', args)
    device = torch.device(args.device)

    dataset_manager = DatasetManager(LABEL_FILE, DATA_DIR)
    ddm = DataDistributionManager(args, dataset_manager, data_dir=DATA_DIR)
    rank = ddm.rank

    train_loader, test_loader = ddm.get_data_loader()

    model, input_shape = get_model(args, device)
    saver = ModelSaver(model, RESULT_DIR, args)
    current_epoch = load_model_weights(saver)

    if rank == 0:
        summary(model, input_shape, device=device.type)

    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=leaning_rate)

    if args.model.lower() != 'dnp':
        train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, saver, device, args.epochs,
                                 rank,
                                 args.world_size)
    else:
        selected_features = train_and_evaluate_dnp(model, train_loader, test_loader, input_shape[0], criterion,
                                                   optimizer, device,
                                                   args.epochs, rank, args.world_size)
    if rank == 0:
        final_evaluation(model, test_loader, device, saver, RESULT_DIR)
