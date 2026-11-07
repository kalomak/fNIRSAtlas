import numpy as np
import pytest
import torch

from fnirsatlas.dl.fNIRSNet23 import fNIRSNet23


def test_fnirsnet23_forward():
    """Test fNIRSNet23 forward."""
    n, ct, c, t = 32, 2, 24, 76
    mock_data = torch.randn(n, ct, c, t)
    net = fNIRSNet23(c, t, ct)
    net.forward(mock_data)


def test_fnirsnet23_forward_batch_size_one():
    """Batch size 1 should still produce logits in training mode."""
    n, ct, c, t = 1, 2, 24, 76
    mock_data = torch.randn(n, ct, c, t)
    net = fNIRSNet23(c, t, ct)
    net.train()

    output = net.forward(mock_data)

    assert output.shape == (n, 2)


def test_fnirsnet23_forward_batch_size_one_eval():
    """Batch size 1 should still produce logits in eval mode."""
    n, ct, c, t = 1, 2, 24, 76
    mock_data = torch.randn(n, ct, c, t)
    net = fNIRSNet23(c, t, ct)
    net.eval()

    with torch.inference_mode():
        output = net.forward(mock_data)

    assert output.shape == (n, 2)


def test_fnirsnet23_fit_last_batch_size_one():
    """Training should handle a final singleton batch."""
    n, ct, c, t = 33, 2, 24, 76
    X = torch.randn(n, ct, c, t)
    y = torch.randint(0, 2, (n,))
    net = fNIRSNet23(c, t, ct, batch_size=32, n_epochs=1)

    net.fit(X, y)

    assert net.is_fitted_


@pytest.mark.slow
def test_fnirsnet23_full():
    """Test fNIRSNet23 full pipe."""
    from fnirsatlas.classification import classify_task
    from fnirsatlas.config import get_classification_tasks
    from fnirsatlas.dl.fNIRSNet23 import create_fnirsnet23_pipeline
    from fnirsatlas.processed_dataset import load_processed_dataset

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
        get_clf=create_fnirsnet23_pipeline,
    )
    print(f"r {r}")
