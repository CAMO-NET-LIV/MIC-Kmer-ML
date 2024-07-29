import datetime
import math
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  #
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_squared_error
import argparse
from dataset_manager import DatasetManager
from models.cnn import CNN
from models.cnn2 import CNN2
from models.kwise import KWise
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
parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
parser.add_argument('--device', type=str, default='cpu', help='Device to use')
parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Master address')
parser.add_argument('--master-port', type=str, default='12345', help='Master port')
parser.add_argument('--world-size', type=int, default=1, help='World size')
parser.add_argument('--timeout', type=int, default=300, help='Timeout for the distributed setup')
parser.add_argument('--in-mem', type=int, default=0, help='Load data in memory')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
args = parser.parse_args()

# Debugging: Print arguments
print('Arguments:', args)

# Initialize the distributed environment if world_size > 1
if args.world_size > 1:
    torch.distributed.init_process_group(
        backend='nccl' if args.device == 'cuda' else 'gloo',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.world_size,
        rank=int(os.environ['SLURM_PROCID']),
        timeout=datetime.timedelta(seconds=args.timeout)
    )

    rank = dist.get_rank()  # Get the rank of the current process

    # Debugging: Confirm initialization
    print('Distributed environment initialized with the following parameters:')
    print(f'MASTER_ADDR: {args.master_addr}')
    print(f'MASTER_PORT: {args.master_port}')
    print(f'WORLD_SIZE: {args.world_size}')
else:
    rank = 0

# Set device
device = torch.device(args.device)

# Configuration constants
LABEL_FILE = 'labels-cleaned.csv'
DATA_DIR = f'../volatile/genome-data-ignore/processed_{args.kmer}mer_count/'
RESULT_DIR = '../volatile/results/'
BATCH_SIZE = args.batch_size
EPOCHS = 1500
leaning_rate = args.lr * args.world_size
SCALE = False

dataset_manager = DatasetManager(LABEL_FILE, DATA_DIR)
train_files, test_files, train_labels, test_labels = dataset_manager.prepare_train_test_path()

train_dataset = CustomDataset(
    file_names=train_files,
    labels=train_labels,
    data_dir=DATA_DIR,
    scale_label=SCALE,
)
test_dataset = CustomDataset(
    file_names=test_files,
    labels=test_labels,
    data_dir=DATA_DIR,
    scale_label=SCALE,
)

if args.in_mem:
    print('Loading data in memory')
    x_train, y_train = zip(
        *[(train_dataset[i][0].squeeze().tolist(), train_dataset[i][1]) for i in range(len(train_files))])
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    x_test, y_test = zip(*[(test_dataset[i][0].squeeze().tolist(), test_dataset[i][1]) for i in range(len(test_files))])
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

# Create distributed samplers and data loaders
if args.world_size > 1:
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
else:
    train_sampler = None  # Use default sampler
    test_sampler = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16, sampler=test_sampler)

# Initialize the model, loss function, and optimizer
model_name = args.model.lower()
if model_name == 'cnn':
    model = CNN(input_dim=4 ** args.kmer, device=device)
    input_shape = (1, 4 ** args.kmer)
elif model_name == 'cnn2':
    model = CNN2(input_dim=4 ** args.kmer, device=device)
    input_shape = (1, 4 ** args.kmer)
elif model_name == 'mlp':
    model = MLP(input_dim=4 ** args.kmer)
    input_shape = (4 ** args.kmer,)
elif model_name == 'kw':
    model = KWise(input_dim=4 ** args.kmer, device=device)
    input_shape = (4 ** args.kmer,)
else:
    raise Exception('Invalid model type. Choose between "CNN" and "MLP".')

saver = ModelSaver(model, RESULT_DIR, model_name, args.kmer, args.device, BATCH_SIZE, EPOCHS)
try:
    current_epoch = saver.load_weight()
    print(f'Model loaded from epoch {current_epoch}')
except FileNotFoundError:
    current_epoch = 0
    print('No saved model found. Training from scratch.')
except Exception as e:
    print(f'Error loading model: {e}')
    current_epoch = 0

model = model.to(device)
print(f'running on: {device}')
if rank == 0:
    summary(model, input_shape, device=device.type)

# Wrap the model with DistributedDataParallel
if args.world_size > 1:
    model = nn.parallel.DistributedDataParallel(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=leaning_rate)  # Scale learning rate

# Training loop
for epoch in range(current_epoch, EPOCHS):
    model.train()
    if args.world_size > 1:
        train_sampler.set_epoch(epoch)  # Set epoch for shuffling the data
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # save model weights every epoch
    if rank == 0:  # Only save weights from the master node
        # Evaluation on training data
        train_rmse, train_ea = evaluate_model(model, train_loader, device, scale=SCALE)
        # Evaluation on testing data
        test_rmse, test_ea = evaluate_model(model, test_loader, device, scale=SCALE)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], '
              f'Train RMSE: {train_rmse:.8f}, Train EA: {train_ea:.8f}, '
              f'Test RMSE: {test_rmse:.8f}, Test EA: {test_ea:.8f}')

        saver.save_weight(epoch)

# Final Evaluation
if rank == 0:  # Only the master node performs the final evaluation and saves results
    final_rmse, final_ea = evaluate_model(model, test_loader, device, scale=SCALE)
    print(f'Final RMSE: {final_rmse:.8f}, Final EA: {final_ea:.8f}')

    predictions, actuals = model_inference(model, test_loader, device)
    ea = is_essential_agreement(actuals, predictions)

    dataframe = {
        'actual': actuals,
        'predicted': predictions,
        'ea': ea
    }

    df = pd.DataFrame(dataframe)
    df.to_csv(os.path.join(RESULT_DIR, saver.get_file_name() + '.csv'), index=False)
