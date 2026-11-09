"""
Benchmark the BNCI dataset
==========================

Run a minimal fNIRS classification benchmark.
"""

from fnirsatlas import fnirs_benchmark
from fnirsatlas.pipelines.defaults import make_fnirs_lda_pipeline

config = {
    "name": "Windowed + LDA",
    "paradigms": ["FNIRSClassification"],
    "pipeline": make_fnirs_lda_pipeline(),
}
results = fnirs_benchmark(
    pipelines=[config],
    evaluations=["CrossSession", "CrossSubject"],
    include_datasets=["Bnci2014FnirsMA"],
)
summary = results.groupby("evaluation")["score"].agg(["mean", "std"])
print(
    (
        summary["mean"].map("{:.1%}".format)
        + " +- "
        + summary["std"].map("{:.1%}".format)
    ).to_string()
)
