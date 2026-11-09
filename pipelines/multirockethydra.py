"""MultiRocketHydra benchmark pipeline."""

from aeon.classification.convolution_based import MultiRocketHydraClassifier
from sklearn.pipeline import make_pipeline

from fnirsatlas.pipelines import AeonClassifier, ChannelMerger, StandardScalerND

PIPELINE = {
    "name": "MultiRocketHydra",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "aeonclassifier__estimator__n_kernels": [2, 4, 8, 32],
        "aeonclassifier__estimator__n_groups": [8, 16, 64, 128],
    },
    "pipeline": make_pipeline(
        ChannelMerger(),
        StandardScalerND(),
        AeonClassifier(MultiRocketHydraClassifier()),
    ),
}
