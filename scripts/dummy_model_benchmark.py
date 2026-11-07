#!/usr/bin/env python3
"""
Runs a dummy model (random classifier) on the full benchmark suite.

After running this script, compare to e.g. the LDA baseline:

```bash
python scripts/compare_models.py --models dummy lda
```

"""

import logging
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fnirsatlas.feature_extraction import WindowedFeatureExtractor
from fnirsatlas.classification import run_benchmark
from fnirsatlas.logging import setup_logging
from fnirsatlas.results import Results


setup_logging()
logger = logging.getLogger(__name__)


def get_dummy_model(**kwargs):
    """Create a dummy model pipeline."""
    random_state = kwargs["random_state"]
    return Pipeline(
        [
            (
                "extractor",
                WindowedFeatureExtractor(n_windows=6, features=["mean"]),
            ),
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(random_state=random_state)),
        ]
    )


def main():
    """Run dummy model benchmark across all datasets and seeds."""
    seeds = list(range(5))
    settings = ["cross", "within"]
    override = True
    for classification_setting in settings:
        logger.info(f"--- Setting: {classification_setting} ---")
        for seed in seeds:
            logger.info(f"Running dummy model benchmark (seed={seed})...")
            results = run_benchmark(
                get_clf=get_dummy_model,
                classification_setting=classification_setting,
                seed=seed,
            )
            results["model_name"] = "dummy"
            save_info = Results.save_to_separate_files(
                results, classification_setting, seeds=[seed], override=override
            )
            logger.info(f"Seed {seed} saved: {save_info}")


if __name__ == "__main__":
    main()
