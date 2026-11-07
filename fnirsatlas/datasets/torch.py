from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .load import load_dataset
from fnirsatlas.io import fNIRSAtlasDataset


class NIRSDataset(Dataset):
    """Thin PyTorch-style wrapper over load_dataset.

    Usage:
        ds = NIRSDataset("bnci")       # by name
        ds = NIRSDataset("/path/file_processed.npz")
    """

    def __init__(
        self,
        dataset: str | Path | fNIRSAtlasDataset | None = None,
    ) -> None:
        if dataset is None:
            raise ValueError("dataset must be provided for NIRSDataset")
        if isinstance(dataset, fNIRSAtlasDataset):
            X, y, groups, _ = dataset.as_tuple()
        else:
            X, y, groups, _ = load_dataset(dataset)
        self._X = X
        self._y = y.astype(np.int64)
        self._groups = groups

    def __len__(self) -> int:
        return int(self._X.shape[0])

    def __getitem__(self, index: int):
        x = self._X[index]
        y = self._y[index]
        g = self._groups[index]
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
            g,
        )

    def dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._X.shape


def make_single_dataset_class(dataset_name: str, class_name: str | None = None):
    """Factory to create a torch Dataset class bound to a specific dataset name.

    Example:
        BNCI = make_single_dataset_class("bnci")
        ds = BNCI(limit=16)
    """

    class _BoundSingleDataset(NIRSDataset):
        def __init__(
            self,
            dataset: str | Path | None = None,
        ) -> None:
            super().__init__(
                dataset=dataset or dataset_name,
            )

    _BoundSingleDataset.__name__ = class_name or dataset_name
    return _BoundSingleDataset
