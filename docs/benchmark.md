# Benchmarking

`fnirs_benchmark` runs sklearn-compatible [pipelines](pipelines.md) across [fNIRS paradigms](paradigms.md) and [datasets](generated/datasets.rst) with [MOABB evaluations](https://moabb.neurotechx.com/docs/api.html#evaluations). It resumes saved results, prints a summary, and returns all result rows as a pandas DataFrame.

## First benchmark

Run a cross-subject benchmark on the BNCI mental-arithmetic dataset with a minimal [sklearn pipeline](pipelines.md):

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import WindowedFeatureExtractor, fnirs_benchmark

pipeline = {
    "name": "LDA",
    "paradigms": ["FNIRSClassification"],
    "pipeline": make_pipeline(
        WindowedFeatureExtractor(),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
}
results = fnirs_benchmark(
    pipelines=[pipeline],
    paradigms=["FNIRSClassification"],
    evaluations=["CrossSubject"],
    include_datasets=["Bnci2014FnirsMA"],
    n_jobs=1,
)
```

## Caching

Downloaded source data and completed evaluation results are cached by default. Preprocessed epochs and arrays are cached only when enabled with `cache_config`.
See [MOABB's cache documentation](https://moabb.neurotechx.com/docs/auto_examples/data_management_and_configuration/plot_disk_cache.html).

## Results and resume

By default, MOABB evaluation results are persisted under MOABB's results directory (`MOABB_RESULTS`, by default `~/mne_data`) and analysis output under `<results>/benchmark`. Change these locations with `results` and `output`:

```python
results = fnirs_benchmark(
    pipelines=[pipeline],
    paradigms=["FNIRSClassification"],
    evaluations=["CrossSubject"],
    include_datasets=["Bnci2014FnirsMA"],
    results="results/experiment-1",
    output="benchmark/experiment-1",
)
```

Matching completed runs are resumed. Set `overwrite=True` to recompute them. `random_state` seeds the cross-validation folds and is set on every pipeline estimator that exposes a `random_state` parameter, so pipeline definitions should not carry one. The seed is included in the result suffix, so runs with different `random_state` values remain separate. Use `suffix` to distinguish other changes that are not represented in the pipeline or evaluation configuration.

See [Statistical analysis](statistical_analysis.md) for further analysis.

## Metrics

Every evaluation records accuracy, Cohen's kappa, and macro f1, precision and
recall. Results carry one `score_<name>` column per metric, and `score` repeats
accuracy. Pass `scorer` in the paradigm context to record something else
instead.

## Multiple seeds

Run each seed separately, then average matching rows:

```python
from fnirsatlas import aggregate_seed_results, fnirs_benchmark

runs = [
    fnirs_benchmark(
        pipelines=[pipeline],
        paradigms=["FNIRSClassification"],
        evaluations=["CrossSubject"],
        include_datasets=["Bnci2014FnirsMA"],
        random_state=seed,
    )
    for seed in (1, 2, 3)
]
results = aggregate_seed_results(runs)
```

## Command line

This script exposes the same common selections for shell workflows:

```bash
uv run python scripts/run_fnirs_benchmark.py --help
```

## Learning curves

Score against the number of training trials per class, with MOABB's
`LearningCurveSplitter`:

```bash
uv run python scripts/learning_curve.py --datasets Bak2019FnirsTapping \
    --sizes 5 10 15 --n-perms 10
```
