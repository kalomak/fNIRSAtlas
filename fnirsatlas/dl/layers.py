import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrintShapeLayer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        logger.debug(f"{self.name:<20} shape: {list(x.shape)}")
        return x
