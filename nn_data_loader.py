import os

from torch.utils.data import Dataset
import numpy as np
from scipy import sparse


class CustomDataset(Dataset):
    def __init__(
            self,
            file_names: [str],
            labels: [float],
            data_dir: str
    ):
        self.files_path = file_names
        self.labels = np.array(labels, dtype=np.float32)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_name = self.files_path[idx]
        path = os.path.join(self.data_dir, file_name)

        content = sparse.load_npz(path).astype(np.float32)
        # convert to dense
        content = content.toarray()
        return content, self.labels[idx]
