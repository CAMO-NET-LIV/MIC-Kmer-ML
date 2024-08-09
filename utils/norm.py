import torch


def lq_norm(x, power=2):
    po = torch.pow(torch.abs(x), power)
    s = torch.sum(po, dim=0)
    return torch.pow(s, 1 / power)


def lp_norm(x, power=2):
    po = torch.pow(torch.abs(x), power)
    s = torch.sum(po)
    return torch.pow(s, 1 / power)


def gradients_batch_norm(x_train, gradients):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    return (gradients - mean) / std
