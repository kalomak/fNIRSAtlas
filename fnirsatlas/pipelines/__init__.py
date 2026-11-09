"""fnirsatlas pipelines."""

from .aeon import AeonClassifier
from .transformations import (
    ChannelMerger,
    ChannelSplitter,
    StandardScalerND,
    UnsqueezeLayer,
    WindowedFeatureExtractor,
)

__all__ = [
    "AeonClassifier",
    "ChannelMerger",
    "ChannelSplitter",
    "StandardScalerND",
    "UnsqueezeLayer",
    "WindowedFeatureExtractor",
]
