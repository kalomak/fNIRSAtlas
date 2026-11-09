"""fNIRSNet24 benchmark pipeline."""

from fnirsatlas.pipelines.dl import fNIRSNet24, make_dl_pipeline

PIPELINE = {
    "name": "fNIRSNet24",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__p_dropout": [0.25, 0.5],
        "fnirsclassifier__module__temporal_filters": [4, 20],
        "fnirsclassifier__module__spatial_filters": [4, 60],
    },
    "pipeline": make_dl_pipeline(fNIRSNet24),
}
