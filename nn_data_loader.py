import os
from typing import Union

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
            scale_label: bool = False,
            scale_input: Union[float, int] = 1,
            sub_features: [int] = None
    ):
        self.files_path = file_names
        self.labels = np.array(labels, dtype=np.float32)
        self.data_dir = data_dir
        self.scale_label = scale_label
        self.scale_input = scale_input
        self.sub_features = sub_features

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_name = self.files_path[idx]
        path = os.path.join(self.data_dir, file_name)

        content = sparse.load_npz(path)
        # convert to dense
        content = content.toarray()
        content = np.asarray(content, dtype=np.float32).reshape(-1)

        if self.sub_features is not None:
            content = content[self.sub_features]

        if self.scale_input != 1:
            content = content / self.scale_input

        if self.scale_label:
            labels = scale_labels(self.labels[idx], min=-3, max=8)
        else:
            labels = self.labels[idx]

        return content, labels
