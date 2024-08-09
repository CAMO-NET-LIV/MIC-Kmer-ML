import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from nn_data_loader import CustomDataset


class DataDistributionManager:
    def __init__(
            self,
            args,
            dataset_manager,
            data_dir,
            min_max_scaler=False,
            input_scale=1,
    ):
        self.args = args
        self.data_dir = data_dir
        self.dataset_manager = dataset_manager
        self.min_max_scaler = min_max_scaler
        self.input_scale = input_scale
        self.batch_size = args.batch_size
        self.rank = self._initialize_distributed(args)

    def _initialize_distributed(self, args):
        if args.world_size > 1:
            torch.distributed.init_process_group(
                backend='nccl' if args.device == 'cuda' else 'gloo',
                init_method=f'tcp://{args.master_addr}:{args.master_port}',
                world_size=args.world_size,
                rank=int(os.environ['SLURM_PROCID']),
                timeout=datetime.timedelta(seconds=args.timeout)
            )
            rank = dist.get_rank()
            print('Distributed environment initialized with the following parameters:')
            print(f'MASTER_ADDR: {args.master_addr}')
            print(f'MASTER_PORT: {args.master_port}')
            print(f'WORLD_SIZE: {args.world_size}')
        else:
            rank = 0
        return rank

    def get_data_loader(self, sub_features=None):
        train_dataset, test_dataset = self._prepare_datasets(self.args, self.dataset_manager, sub_features)
        train_loader, test_loader = self._create_data_loaders(train_dataset, test_dataset, self.batch_size, self.args.world_size)
        return train_loader, test_loader


    def _load_data_in_memory(self, dataset, files):
        data, labels = zip(*[(dataset[i][0].squeeze().tolist(), dataset[i][1]) for i in range(len(files))])
        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return TensorDataset(torch.from_numpy(data), torch.from_numpy(labels))


    def _prepare_datasets(self, args, dataset_manager, sub_features):
        train_files, test_files, train_labels, test_labels = dataset_manager.prepare_train_test_path()

        train_dataset = CustomDataset(
            file_names=train_files,
            labels=train_labels,
            data_dir=self.data_dir,
            scale_label=self.min_max_scaler,
            scale_input=self.input_scale,
            sub_features=sub_features,
        )
        test_dataset = CustomDataset(
            file_names=test_files,
            labels=test_labels,
            data_dir=self.data_dir,
            scale_label=self.min_max_scaler,
            scale_input=self.input_scale,
            sub_features=sub_features,
        )

        if args.in_mem:
            print('Loading data in memory')
            train_dataset = self._load_data_in_memory(train_dataset, train_files)
            test_dataset = self._load_data_in_memory(test_dataset, test_files)

        return train_dataset, test_dataset


    def _create_data_loaders(self, train_dataset, test_dataset, batch_size, world_size):
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        else:
            train_sampler = None
            test_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, sampler=test_sampler)
        return train_loader, test_loader
