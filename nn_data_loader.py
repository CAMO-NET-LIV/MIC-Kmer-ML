import os

from torch.utils.data import Dataset
import numpy as np
from scipy import sparse

from utils.scale import scale_labels


class CustomDataset(Dataset):
    def __init__(
            self,
            file_names: [str],
            labels: [float],
            data_dir: str,
            scale_label: bool = False
    ):
        self.files_path = file_names
        self.labels = np.array(labels, dtype=np.float32)
        self.data_dir = data_dir
        self.scale_label = scale_label

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_name = self.files_path[idx]
        path = os.path.join(self.data_dir, file_name)

        content = sparse.load_npz(path)
        # convert to dense
        content = content.toarray()
        content = np.asarray(content, dtype=np.float32)
        if self.scale_label:
            labels = scale_labels(self.labels[idx], min=-3, max=8)
        else:
            labels = self.labels[idx]

        return content, labels

