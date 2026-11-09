"""Locate and load benchmark results."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from moabb.analysis.results import Results as MoabbResults

from fnirsatlas.benchmark import (
    EVALUATIONS,
    PARADIGMS,
    _seed_suffix,
    results_dir,
    select_datasets,
)


def cell_suffix(dataset: str, pipeline: str, seed: int | str) -> str:
    """Return the MOABB results suffix of one benchmark cell."""
    return _seed_suffix(f"{dataset}_{pipeline}", seed)


def result_path(
    paradigm: str,
    evaluation: str,
    dataset: str,
    pipeline: str,
    seed: int | str,
    root: Path | None = None,
) -> Path:
    """Return the MOABB result file of one benchmark cell.

    Parameters
    ----------
    paradigm, evaluation, dataset, pipeline : str
        Paradigm class name, evaluation name without ``Evaluation``, dataset
        code and pipeline file stem.
    seed : int or str
        Random seed of the run.
    root : Path or None
        Results root. None uses :func:`fnirsatlas.benchmark.results_dir`.

    Returns
    -------
    Path
        The file MOABB ``Results`` writes for this cell.
    """
    return (
        (root or results_dir())
        / "results"
        / paradigm
        / f"{evaluation}Evaluation"
        / f"results_{cell_suffix(dataset, pipeline, seed)}.hdf5"
    )


def load_results(
    paradigm: str,
    evaluation: str,
    pipelines: Sequence[str],
    seeds: Sequence[int],
    datasets: Sequence[str] | None = None,
    exclude_datasets: Sequence[str] | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load the benchmark results of pipelines on the datasets of a paradigm.

    Parameters
    ----------
    paradigm : str
        Paradigm class name, e.g. ``FNIRSClassification``.
    evaluation : str
        Evaluation name without ``Evaluation``, e.g. ``CrossSubject``.
    pipelines : sequence of str
        Pipeline file stems, e.g. ``Windowed_LDA``.
    seeds : sequence of int
        Seeds to load. Rows of all seeds are returned.
    datasets, exclude_datasets : sequence of str or None
        Dataset codes to keep or drop. None keeps every dataset of the paradigm.
    root : Path or None
        Results root. None uses :func:`fnirsatlas.benchmark.results_dir`.

    Returns
    -------
    pandas.DataFrame
        One row per score, with ``pipeline`` set to the pipeline file stem and a
        ``seed`` column. Raises if any result file is missing.
    """
    root = root or results_dir()
    selected = select_datasets(PARADIGMS[paradigm](), datasets, exclude_datasets)
    cells = [
        (dataset.code, pipeline, seed)
        for dataset in selected
        for pipeline in pipelines
        for seed in seeds
    ]
    # MOABB Results creates an empty file for a missing cell, so check first.
    paths = [result_path(paradigm, evaluation, *cell, root) for cell in cells]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} result files missing: {missing}")
    frames = [
        MoabbResults(
            EVALUATIONS[evaluation],
            PARADIGMS[paradigm],
            suffix=cell_suffix(code, pipeline, seed),
            hdf5_path=str(root),
        )
        .to_dataframe()
        .assign(pipeline=pipeline, seed=seed)
        for code, pipeline, seed in cells
    ]
    return pd.concat(frames, ignore_index=True)


def subject_mean(
    results: pd.DataFrame, keys: Sequence[str], score: str = "score"
) -> pd.Series:
    """Return the mean score per group with every subject weighted equally.

    Parameters
    ----------
    results : pandas.DataFrame
        Result rows with ``subject`` and ``score`` columns.
    keys : sequence of str
        Columns that define a group, e.g. ``["dataset", "pipeline"]``.
    score : str
        Score column, e.g. ``score_kappa``.

    Returns
    -------
    pandas.Series
        Mean over subjects of each subject's mean over its rows, such as
        sessions and seeds, indexed by ``keys``.
    """
    keys = list(keys)
    return results.groupby([*keys, "subject"])[score].mean().groupby(keys).mean()


def score_matrix(results: pd.DataFrame, score: str = "score") -> pd.DataFrame:
    """Return the mean score of each pipeline on each dataset in percent.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`load_results`.
    score : str
        Score column, e.g. ``score_kappa``.

    Returns
    -------
    pandas.DataFrame
        Datasets as rows, pipelines as columns, from :func:`subject_mean`.
    """
    return subject_mean(results, ["dataset", "pipeline"], score).unstack().mul(100)
