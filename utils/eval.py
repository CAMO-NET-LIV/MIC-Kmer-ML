import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import pandas as pd

from util import is_essential_agreement


def evaluate_model(model, data_loader, device):
    predictions, actuals = model_inference(model, data_loader, device)

    mse_loss = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse_loss)
    ea = np.mean(is_essential_agreement(actuals, predictions))
    return rmse, ea


def model_inference(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().squeeze().numpy())
            actuals.extend(batch_y.cpu().squeeze().numpy())

    return predictions, actuals
