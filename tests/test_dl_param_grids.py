"""Test that every DL model runs a forward pass with all param grid settings."""

import torch
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from fnirsatlas.config import get_param_grid
from fnirsatlas.dl.CNN2a import CNN2a
from fnirsatlas.dl.EEGNet import EEGNet
from fnirsatlas.dl.DCN import DCN
from fnirsatlas.dl.fNIRSNet23 import fNIRSNet23
from fnirsatlas.dl.fNIRSNet24 import fNIRSNet24
from fnirsatlas.dl.transformers import fNIRS_T

N, CT = 4, 2
SPLIT_EDGE_SHAPES = [(4, 45), (100, 200)]
FNIRST_EDGE_SHAPES = [(5, 45), (100, 200)]


def _strip_prefix(grid, prefix="clf__"):
    """Strip pipeline prefix from param grid keys."""
    return {k.replace(prefix, ""): v for k, v in grid.items()}


def _forward_with_params(
    model_cls, base_kwargs_fn, param_grid, input_shape_fn=None, shapes=SPLIT_EDGE_SHAPES
):
    """Verify forward pass for every combination in the param grid."""
    grid = _strip_prefix(param_grid)
    for n_chs, T in shapes:
        input_shape = input_shape_fn(n_chs, T) if input_shape_fn else (N, CT, n_chs, T)
        x = torch.randn(*input_shape)
        for params in ParameterGrid(grid):
            net = model_cls(**base_kwargs_fn(n_chs, T))
            net.set_params(**params)
            net = clone(net)
            out = net(x)
            assert out.shape[0] == input_shape[0], (
                f"Bad output for shape={(n_chs, T)} {params}: {out.shape}"
            )


def test_cnn2a_all_grid_settings():
    grid = get_param_grid("cnn2a")
    assert grid, "No param grid found for cnn2a"
    _forward_with_params(
        CNN2a,
        lambda n_chs, T: {"n_chs": n_chs, "T": T, "n_ch_types": CT},
        grid,
        shapes=SPLIT_EDGE_SHAPES,
    )


def test_eegnet_all_grid_settings():
    grid = get_param_grid("eegnet")
    assert grid, "No param grid found for eegnet"
    _forward_with_params(
        EEGNet,
        lambda n_chs, T: {"n_chs": n_chs, "T": T, "n_ch_types": CT},
        grid,
        shapes=SPLIT_EDGE_SHAPES,
    )


def test_dcn_all_grid_settings():
    grid = get_param_grid("dcn")
    assert grid, "No param grid found for dcn"
    _forward_with_params(
        DCN,
        lambda n_chs, T: {"n_chs": n_chs, "T": T, "n_ch_types": CT},
        grid,
        shapes=SPLIT_EDGE_SHAPES,
    )


def test_fnirst_all_grid_settings():
    grid = get_param_grid("fnirst")
    assert grid, "No param grid found for fnirst"
    _forward_with_params(
        fNIRS_T,
        lambda n_chs, T: {"n_class": 2, "sampling_point": T, "n_ch_types": CT},
        grid,
        shapes=FNIRST_EDGE_SHAPES,
    )


def test_fnirsnet23_all_grid_settings():
    grid = get_param_grid("fnirsnet23")
    assert grid, "No param grid found for fnirsnet23"
    _forward_with_params(
        fNIRSNet23,
        lambda n_chs, T: {"n_chs": n_chs, "T": T, "n_ch_types": CT},
        grid,
        shapes=SPLIT_EDGE_SHAPES,
    )


def test_fnirsnet24_all_grid_settings():
    grid = get_param_grid("fnirsnet24")
    assert grid, "No param grid found for fnirsnet24"
    _forward_with_params(
        fNIRSNet24,
        lambda n_chs, T: {"n_chs": n_chs * CT, "T": T},
        grid,
        input_shape_fn=lambda n_chs, T: (N, 1, n_chs * CT, T),
        shapes=SPLIT_EDGE_SHAPES,
    )
