"""Behavioral tests for the dual-branch fNIRS transformer."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from fnirsatlas.pipelines.dl import fNIRS_T


@pytest.mark.parametrize("n_channels", [1, 4, 104])
@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_transformer_supports_channel_counts_and_singleton_batches(n_channels, pool):
    model = fNIRS_T(
        n_channels=n_channels,
        n_times=30,
        n_outputs=3,
        dim=8,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=8,
        pool=pool,
    )
    logits = model(torch.randn(1, 2, n_channels, 30))
    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
    logits.square().sum().backward()
    assert model.cls_token_patch.grad is not None
    assert model.cls_token_channel.grad is not None


def test_transformer_rejects_epochs_shorter_than_temporal_kernel():
    with pytest.raises(ValueError, match="at least 30"):
        fNIRS_T(n_channels=4, n_times=29, n_outputs=2)
