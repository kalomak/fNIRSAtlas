"""EEGNet benchmark pipeline."""

from fnirsatlas.pipelines.dl import EEGNet, make_dl_pipeline

PIPELINE = {
    "name": "EEGNet",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__dropout": [0.25, 0.5],
        "fnirsclassifier__module__F1": [4, 16],
        "fnirsclassifier__module__kernel_size": [3, 10],
    },
    "pipeline": make_dl_pipeline(EEGNet),
}
