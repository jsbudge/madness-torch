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
        self.data = torch.tensor(Xt.values) if is_val else torch.tensor(Xs.values)

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
