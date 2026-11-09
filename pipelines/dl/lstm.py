"""LSTM benchmark pipeline."""

from fnirsatlas.pipelines.dl import LSTM, make_dl_pipeline

PIPELINE = {
    "name": "LSTM",
    "paradigms": ["FNIRSClassification"],
    "param_grid": {
        "fnirsclassifier__lr": [0.01, 0.001],
        "fnirsclassifier__module__hidden_size": [32, 64],
        "fnirsclassifier__module__num_layers": [2, 4],
        "fnirsclassifier__module__p_dropout": [0.25, 0.5],
    },
    "pipeline": make_dl_pipeline(LSTM),
}
