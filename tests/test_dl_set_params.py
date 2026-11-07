"""Test that clone after set_params rebuilds model architecture."""

import torch
from sklearn.base import clone
from fnirsatlas.dl.EEGNet import EEGNet


def _get_conv_kernel_size(net):
    """Get kernel_size of the first Conv2d in firstConv."""
    for m in net.firstConv:
        if isinstance(m, torch.nn.Conv2d):
            return m.kernel_size
    raise ValueError("No Conv2d found in firstConv")


def test_eegnet_clone_after_set_params_rebuilds_layers():
    """clone() after set_params must produce a network with updated layers."""
    n_chs, T, n_ch_types = 4, 64, 2
    net = EEGNet(n_chs, T, n_ch_types, kernel_size_1=3)
    assert _get_conv_kernel_size(net) == (1, 3)
    net.set_params(kernel_size_1=10)
    net = clone(net)
    assert net.kernel_size_1 == 10
    assert _get_conv_kernel_size(net) == (1, 10), (
        "clone after set_params did not rebuild layers with new kernel_size_1"
    )
