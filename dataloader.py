from glob import glob
from typing import List, Optional, Union, Iterator
import os
import yaml
from load_data import load
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

from load_data import getMatches, normalize, prepFrame


def gd_collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]), torch.tensor([b[4] for b in batch]))


class GameDataset(Dataset):
    def __init__(self, datapath: str = './data', season: int = None, is_val: bool = False, seed: int = 7):
        # Load in data
        self.datapath = datapath
        self.data = []
        '''for s in range(2004, 2025):
            dp = f'{datapath}/t{s}' if not is_val else f'{datapath}/{s}'
            if Path(dp).exists():
                self.data.append(glob(f'{dp}/*.pt'))'''
        if season is None:
            start = 2021 if is_val else 2004
            end = 2025 if is_val else 2021
            for s in range(start, end):
                dp = f'{datapath}/t{s}'
                if Path(dp).exists():
                    self.data.append(glob(f'{dp}/*.pt'))
        else:
            for s in range(2004, 2025):
                if s == season and is_val or s != season and not is_val:
                    dp = f'{datapath}/t{s}'
                    if Path(dp).exists():
                        self.data.append(glob(f'{dp}/*.pt'))
        self.data = np.concatenate(self.data)
        np.random.shuffle(self.data)
        # Xt, Xs = train_test_split(self.data, random_state=seed)
        # self.data = Xs if is_val else Xt
        check = torch.load(self.data[0])
        self.data_len = check[0].shape[-1]
        self.extra_len = check[2].shape[-1]

    def __getitem__(self, idx):
        return torch.load(self.data[idx])

    def __len__(self):
        return self.data.shape[0]


class GameDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            datapath: str = './data',
            season: int = None,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.datapath = datapath
        self.season = season

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = GameDataset(self.datapath, season=self.season)
        self.val_dataset = GameDataset(self.datapath, season=self.season, is_val=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=gd_collate,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=gd_collate,
        )


class PredictorDataset(Dataset):
    def __init__(self, datapath: str = './data', split: float = 1., fnme: str = 'SimpleAverages', is_val: bool = False, seed: int = 7):
        # Load in data
        self.datapath = datapath
        gids = pd.read_csv(f'{self.datapath}/MNCAATourneyCompactResults.csv')
        gids = prepFrame(gids, True)
        results = pd.DataFrame(data=(gids['t_score'] - gids['o_score']) > 0)
        features = load(fnme, datapath=datapath)
        self.data = getMatches(gids, features)
        self.labels = results
        self.data_len = self.data[0].shape[1]

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.labels[idx]

    def __len__(self):
        return self.data[0].shape[0]

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
            file: str = None,
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
        self.file = file

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = PredictorDataset(self.datapath, self.split, fnme=self.file)
        self.val_dataset = PredictorDataset(self.datapath, self.split, fnme=self.file, is_val=True)

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
