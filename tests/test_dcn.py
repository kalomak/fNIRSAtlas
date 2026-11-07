import numpy as np
import pytest
import torch
from fnirsatlas.dl.DCN import DCN


def test_dcn_forward():
    """Test DCN forward."""
    n, ct, c, t = 32, 2, 24, 76
    mock_data = torch.randn(n, ct, c, t)
    net = DCN(c, t, ct)
    net.forward(mock_data)


@pytest.mark.slow
def test_dcn_full():
    """Test DCN full pipe."""
    from fnirsatlas.processed_dataset import load_processed_dataset
    from fnirsatlas.config import get_classification_tasks
    from fnirsatlas.classification import classify_task
    from fnirsatlas.dl.DCN import create_dcn_pipeline

    dataset_name = "bnci"
    X, y, g, i = load_processed_dataset(dataset_name)
    task = get_classification_tasks(dataset_name, cond2label=i["cond2label"])[0]
    r = classify_task(
        dataset_name,
        task,
        X=X,
        y=y,
        info=i,
        groups=g,
        classification_setting="cross",
        get_clf=create_dcn_pipeline,
    )
    print(f"r {r}")
