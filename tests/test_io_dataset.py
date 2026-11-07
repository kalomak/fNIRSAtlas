import numpy as np

import fnirsatlas.config as cfg
from fnirsatlas.io import fNIRSAtlasDataset, FNIRSAtlasDataset


def test_fnirsatlas_dataset_loads_by_name(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed_datasets"
    processed_dir.mkdir(parents=True, exist_ok=True)

    name = "dummyset"
    dataset_path = processed_dir / f"{name}_processed.npz"
    X = np.random.randn(4, 3, 10)
    y = np.array([0, 1, 0, 1])
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=str)
    info = {"sfreq": 10.0, "ch_names": ["ch1", "ch2", "ch3"]}
    np.savez(dataset_path, X=X, y=y, groups=groups, info=info)

    monkeypatch.setattr(
        cfg,
        "get_processed_datasets_dir",
        lambda: processed_dir,
    )

    ds = fNIRSAtlasDataset(name)
    X2, y2, groups2, info2 = ds.as_tuple()

    assert np.allclose(X2, X)
    assert np.array_equal(y2, y)
    assert np.array_equal(groups2, groups)
    assert info2["sfreq"] == info["sfreq"]


def test_fnirsatlas_dataset_alias():
    assert FNIRSAtlasDataset is fNIRSAtlasDataset


def test_fnirsatlas_dataset_applies_task_mapping(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed_datasets"
    processed_dir.mkdir(parents=True, exist_ok=True)

    name = "dummyset"
    dataset_path = processed_dir / f"{name}_processed.npz"
    X = np.random.randn(5, 3, 10)
    y = np.array([0, 1, 2, 1, 0])
    groups = np.array(["s1", "s1", "s2", "s2", "s3"], dtype=str)
    info = {
        "sfreq": 10.0,
        "ch_names": ["ch1", "ch2", "ch3"],
        "cond2label": {"left": 0, "right": 1, "rest": 2},
    }
    np.savez(dataset_path, X=X, y=y, groups=groups, info=info)

    monkeypatch.setattr(
        cfg,
        "get_processed_datasets_dir",
        lambda: processed_dir,
    )

    monkeypatch.setattr(
        "fnirsatlas.io.dataset.get_classification_tasks",
        lambda dataset_name, cond2label=None: [
            {"description": "left-right", "mapping": {"left": 0, "right": 1}}
        ],
    )

    ds = fNIRSAtlasDataset(name, task="left-right")

    assert ds.task_description == "left-right"
    assert ds.cond2label == {"left": 0, "right": 1}
    assert np.array_equal(ds.y, np.array([0, 1, 1, 0]))
    assert ds.X.shape[0] == 4
    assert ds.groups.shape[0] == 4
