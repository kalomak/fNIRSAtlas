"""
==========================
Benchmark the BNCI dataset
==========================

Run a minimal fNIRS classification benchmark with the default windowed LDA
pipeline on the BNCI mental arithmetic dataset.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from fnirsatlas import fnirs_benchmark
from fnirsatlas.pipelines.defaults import make_fnirs_lda_pipeline

# %%
# Define the pipeline
# -------------------
#
# A pipeline configuration names a scikit-learn pipeline and the paradigms it
# is evaluated on.

config = {
    "name": "Windowed + LDA",
    "paradigms": ["FNIRSClassification"],
    "pipeline": make_fnirs_lda_pipeline(),
}

# %%
# Run the benchmark
# -----------------
#
# The dataset is downloaded on first use. Results are cached, so a second run
# returns immediately. The mean score per evaluation is printed.

results = fnirs_benchmark(
    pipelines=[config],
    evaluations=["WithinSession", "CrossSession", "CrossSubject"],
    include_datasets=["Bnci2014FnirsMA"],
)
