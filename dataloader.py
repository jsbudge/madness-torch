from glob import glob
from typing import List, Optional, Union, Iterator
import os
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split


class EncoderDataset(Dataset):
    def __init__(self, datapath: str = './data', split: float = 1., is_val: bool = False, seed: int = 7):
        # Load in data
        self.datapath = datapath
        csv_df = pd.read_csv(self.datapath)

        if split < 1:
            Xs, Xt, _, _ = train_test_split(csv_df,
                                            csv_df,
                                            test_size=split, random_state=seed)
        else:
            Xt = csv_df
            Xs = csv_df
        self.data = torch.tensor(Xt.values, dtype=torch.float32) if is_val else torch.tensor(Xs.values, dtype=torch.float32)
        self.data_len = self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class EncoderDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            split: float = .7,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 0  # cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.datapath = datapath
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = EncoderDataset(self.datapath, self.split)
        self.val_dataset = EncoderDataset(self.datapath, self.split, is_val=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class PredictorDataset(Dataset):
    def __init__(self, datapath: str = './data', split: float = 1., is_val: bool = False, seed: int = 7):
        # Load in data
        self.datapath = datapath
        t0 = pd.read_csv(f'{self.datapath}/MTrainingData_0.csv').set_index(['gid', 'season', 'tid', 'oid'])
        t1 = pd.read_csv(f'{self.datapath}/MTrainingData_1.csv').set_index(['gid', 'season', 'tid', 'oid'])
        results = pd.read_csv(f'{self.datapath}/MTrainingData_label.csv').set_index(['gid', 'season', 'tid', 'oid'])

        if split < 1:
            Xs0, Xt0, Xs1, Xt1, ys, yt = train_test_split(t0, t1, results, test_size=split, random_state=seed)
        else:
            Xt0 = t0
            Xs0 = t0
            Xt1 = t1
            Xs1 = t1
            ys = results
            yt = results
        self.d0 = torch.tensor(Xt0.values, dtype=torch.float32) if is_val else torch.tensor(Xs0.values, dtype=torch.float32)
        self.d1 = torch.tensor(Xt1.values, dtype=torch.float32) if is_val else torch.tensor(Xs1.values, dtype=torch.float32)
        self.labels = torch.tensor(yt.values, dtype=torch.float32) if is_val else torch.tensor(ys.values, dtype=torch.float32)
        self.data_len = self.d0.shape[1]

    def __getitem__(self, idx):
        return self.d0[idx], self.d1[idx], self.labels[idx]

    def __len__(self):
        return self.d0.shape[0]

class PredictorDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            split: float = .7,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 0  # cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.datapath = datapath
        self.split = split

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PredictorDataset(self.datapath, self.split)
        self.val_dataset = PredictorDataset(self.datapath, self.split, is_val=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
