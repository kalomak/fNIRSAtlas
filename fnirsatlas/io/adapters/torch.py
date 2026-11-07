import numpy as np


def to_torch_dataset(X, y, groups):
    try:
        import torch
        from torch.utils.data import Dataset
    except ModuleNotFoundError as e:
        raise ImportError("PyTorch not available. Install torch to use to_torch().") from e

    y64 = y.astype(np.int64)

    class _TorchAdapterDataset(Dataset):
        def __len__(self):
            return int(X.shape[0])

        def __getitem__(self, index: int):
            return (
                torch.as_tensor(X[index], dtype=torch.float32),
                torch.as_tensor(y64[index], dtype=torch.long),
                groups[index],
            )

    return _TorchAdapterDataset()
