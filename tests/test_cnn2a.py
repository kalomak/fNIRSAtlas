import torch
from fnirsatlas.dl.CNN2a import CNN2a


def test_cnn2a_shapes():
    """Verify CNN2a forward shapes match Table 4 of Trakoolwilaiwan et al. (2017)."""
    T, n_chs, n_ch_types = 257, 34, 2
    model = CNN2a(n_chs, T, n_ch_types, num_classes=3)
    model.eval()

    x = torch.zeros(4, n_ch_types, n_chs, T)
    with torch.no_grad():
        a1 = model.block1(x)
        assert a1.shape == (4, 32, 1, 128), f"After block1: {a1.shape}"
        a2 = model.block2(a1)
        assert a2.shape == (4, 32, 1, 64), f"After block2: {a2.shape}"
        a3 = model.block3(a2)
        assert a3.shape == (4, 32, 1, 32), f"After block3: {a3.shape}"
        assert a3.flatten(1).shape == (4, 1024), f"Flatten: {a3.flatten(1).shape}"
        out = model(x)
        assert out.shape == (4, 3), f"Output: {out.shape}"
