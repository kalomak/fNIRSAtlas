import unittest
import numpy as np
import torch
from fnirsatlas.dl.EEGNet import EEGNet

def test_eegnet_forward():
    """Test eegnet forward."""
    n, ct, c, t = 32, 2, 24, 76
    mock_data = torch.randn(n, ct, c, t)
    net = EEGNet(c, t, ct)
    net.forward(mock_data)

