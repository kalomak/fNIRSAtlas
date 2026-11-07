import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from fnirsatlas.config import get_dataloader_config


def to_dataloader(X, y, batch_size=32, device="cpu", **kwargs):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)
    dl_kwargs = get_dataloader_config(device=device)
    dl_kwargs.update(kwargs)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, **dl_kwargs)
