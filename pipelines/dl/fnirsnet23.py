"""fNIRSNet23 benchmark pipeline."""

from fnirsatlas.pipelines.dl import fNIRSNet23, make_dl_pipeline

PIPELINE = {
    "name": "fNIRSNet23",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__F1": [2, 4, 8],
        "fnirsclassifier__module__F2": [8, 16, 32],
    },
    "pipeline": make_dl_pipeline(fNIRSNet23),
}
