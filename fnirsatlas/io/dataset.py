from pathlib import Path
from typing import Union

import numpy as np

from fnirsatlas.config import get_classification_tasks
from .adapters import to_mne_epochs, to_torch_dataset
from .processed_dataset import load_processed_dataset


def _infer_dataset_name(dataset: Union[str, Path]) -> str:
    name = Path(dataset).name
    if name.endswith("_processed.npz"):
        return name.rsplit("_processed.npz", 1)[0]
    return name


class fNIRSAtlasDataset:
    def __init__(self, dataset: Union[str, Path], task: Union[str, dict, None] = None):
        self.X, self.y, self.groups, self.info = load_processed_dataset(dataset)
        self.dataset_name = _infer_dataset_name(dataset)
        self._X_all = self.X
        self._y_all = self.y
        self._groups_all = self.groups
        self.cond2label = None
        self.task_description = None
        if task is not None:
            self.set_task(task)

    @classmethod
    def from_arrays(cls, X, y, groups, info):
        obj = cls.__new__(cls)
        obj.X = X
        obj.y = y
        obj.groups = groups
        obj.info = info
        obj.dataset_name = ""
        obj._X_all = X
        obj._y_all = y
        obj._groups_all = groups
        obj.cond2label = None
        obj.task_description = None
        return obj

    def get_tasks(self) -> list[dict]:
        return get_classification_tasks(
            self.dataset_name,
            cond2label=self.info.get("cond2label"),
        )

    def get_cond2label(self) -> dict:
        if self.cond2label is not None:
            return self.cond2label
        tasks = self.get_tasks()
        if tasks:
            return tasks[0]["mapping"]
        return self.info.get("cond2label", {})

    def set_task(self, task: Union[str, dict, None] = None):
        selected = task
        if selected is None:
            tasks = self.get_tasks()
            selected = tasks[0] if tasks else {"description": "", "mapping": self.get_cond2label()}
        if isinstance(selected, str):
            by_desc = {t["description"]: t for t in self.get_tasks()}
            if selected not in by_desc:
                raise ValueError(f"Unknown task '{selected}'. Available: {list(by_desc.keys())}")
            selected = by_desc[selected]
        mapping = selected["mapping"] if isinstance(selected, dict) else selected
        desc = selected.get("description", "") if isinstance(selected, dict) else ""
        self._apply_task_mapping(mapping, desc)
        return self

    def _apply_task_mapping(self, cond2label: dict, description: str = ""):
        y = self._y_all
        label2cond = {v: k for k, v in self.info["cond2label"].items()}
        conds = np.array([label2cond[c] for c in y])
        keep_conds = list(cond2label.keys())
        y_mask = np.isin(conds, keep_conds)
        self.X = self._X_all[y_mask]
        self.groups = self._groups_all[y_mask]
        conds = conds[y_mask]
        self.y = np.array([cond2label[c] for c in conds])
        self.cond2label = cond2label
        self.task_description = description

    def as_tuple(self):
        return self.X, self.y, self.groups, self.info

    def to_torch(self):
        return to_torch_dataset(self.X, self.y, self.groups)

    def to_mne(self):
        return to_mne_epochs(self.X, self.y, self.groups, self.info)


FNIRSAtlasDataset = fNIRSAtlasDataset
