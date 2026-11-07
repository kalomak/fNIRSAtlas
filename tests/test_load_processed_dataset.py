import numpy as np
from pathlib import Path
import types

from fnirsatlas.processed_dataset import load_processed_dataset
import fnirsatlas.config as cfg


def test_load_processed_dataset_by_name(tmp_path, monkeypatch):
    # Arrange: create a fake processed dataset under a temp processed dir
    processed_dir = tmp_path / "processed_datasets"
    processed_dir.mkdir(parents=True, exist_ok=True)

    name = "dummyset"
    dataset_path = processed_dir / f"{name}_processed.npz"

    X = np.random.randn(4, 3, 10)
    y = np.array([0, 1, 0, 1])
    groups = np.array(["s1", "s1", "s2", "s2"], dtype=str)
    info = {
        "sfreq": 10.0,
        "ch_names": ["ch1", "ch2", "ch3"],
        "cond2label": {"a": 0, "b": 1},
        "epoch_duration": 1.0,
        "tmin": 0.0,
        "tmax": 1.0,
        "chs": [
            {"coil_type": 0},
            {"coil_type": 0},
            {"coil_type": 0},
        ],
        "montage_positions": None,
    }
    np.savez(dataset_path, X=X, y=y, groups=groups, info=info)

    # Monkeypatch the processed datasets dir to our temp dir
    monkeypatch.setattr(
        cfg,
        "get_processed_datasets_dir",
        lambda: processed_dir,
    )

    # Act
    X2, y2, groups2, info2 = load_processed_dataset(name)

    # Assert
    assert np.allclose(X2, X)
    assert np.array_equal(y2, y)
    assert np.array_equal(groups2, groups)
    assert isinstance(info2, dict)
    assert info2["sfreq"] == info["sfreq"]
