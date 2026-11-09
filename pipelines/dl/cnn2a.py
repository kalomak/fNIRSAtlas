"""CNN2a benchmark pipeline."""

from fnirsatlas.pipelines.dl import CNN2a, make_dl_pipeline

PIPELINE = {
    "name": "CNN2a",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__p_dropout": [0.25, 0.5],
        "fnirsclassifier__module__n_filters": [16, 32],
        "fnirsclassifier__module__kernel_size": [3, 10],
    },
    "pipeline": make_dl_pipeline(CNN2a),
}
