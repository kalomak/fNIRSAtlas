"""fNIRS-T benchmark pipeline."""

from fnirsatlas.pipelines.dl import fNIRS_T, make_dl_pipeline

PIPELINE = {
    "name": "fNIRS-T",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__dim": [64, 128],
        "fnirsclassifier__module__depth": [2, 6],
        "fnirsclassifier__module__dropout": [0.0, 0.3],
    },
    "pipeline": make_dl_pipeline(fNIRS_T),
}
