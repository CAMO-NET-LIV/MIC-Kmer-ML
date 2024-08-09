import torch
from torch import nn
import os


class ModelSaver:
    def __init__(
            self,
            model: nn.Module,
            folder: str,
            args
    ):
        self.model = model
        self.folder = folder
        self.args = args
        self.model_name = args.model.lower()
        self.kmer = args.kmer
        self.device = args.device
        self.batch_size = args.batch_size
        self.epochs = args.epochs

    def save_weight(self, current_epoch: int):
        filepath = self.get_filepath()

        torch.save({
            'epoch': current_epoch,
            'model_state_dict': self.model.state_dict(),
        }, filepath)

    def load_weight(self):
        checkpoint = torch.load(self.get_filepath())
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch']

    def get_file_name(self):
        return f'model_{self.kmer}mer_{self.device}_{self.model_name}_{self.batch_size}batch_{self.epochs}epochs'

    def get_filepath(self):
        return os.path.join(
            self.folder, self.get_file_name()
        ) + '.pt'
