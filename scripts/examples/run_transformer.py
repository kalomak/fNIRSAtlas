from fnirsatlas.classification import run_benchmark
from fnirsatlas.dl.transformers import create_transformer_pipeline

res = run_benchmark(
    get_clf=create_transformer_pipeline,
    suite="ma",
    classification_setting="cross",
    seed=42,
    n_jobs=1,
)
