"""DCN benchmark pipeline."""

from fnirsatlas.pipelines.dl import DCN, make_dl_pipeline

PIPELINE = {
    "name": "DCN",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__p_dropout": [0.25, 0.5],
        "fnirsclassifier__module__D1": [4, 25],
        "fnirsclassifier__module__kernel_size": [2, 10],
    },
    "pipeline": make_dl_pipeline(DCN),
}
