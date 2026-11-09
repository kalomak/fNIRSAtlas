"""Benchmark fNIRS pipelines with MOABB evaluations."""

import logging
from collections.abc import Mapping
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import yaml
from mne import get_config
from moabb.analysis import analyze
from moabb.evaluations import (
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
    WithinSessionEvaluation,
)
from moabb.pipelines.utils import (
    generate_paradigms,
    generate_param_grid,
    parse_pipelines_from_directory,
)

from fnirsatlas import datasets
from fnirsatlas.paradigms import (
    Emotion,
    FingerTapping,
    FNIRSClassification,
    MentalArithmetic,
    MotorExecution,
    MotorImagery,
    NBack,
)

log = logging.getLogger(__name__)

EVALUATIONS = {
    "WithinSession": WithinSessionEvaluation,
    "CrossSession": CrossSessionEvaluation,
    "CrossSubject": CrossSubjectEvaluation,
}
# CrossSession uses LOO so shuffle errors
SHUFFLED_EVALUATIONS = frozenset({"WithinSession", "CrossSubject"})
PARADIGMS = {
    cls.__name__: cls
    for cls in (
        FNIRSClassification,
        Emotion,
        FingerTapping,
        MentalArithmetic,
        MotorExecution,
        MotorImagery,
        NBack,
    )
}
# Cheap baseline used when no definition file is named, so a bare run stays quick.
DEFAULT_PIPELINES = ["pipelines/Windowed_LogReg.yml"]
RESULT_KEYS = [
    "paradigm",
    "evaluation",
    "dataset",
    "subject",
    "session",
    "pipeline",
]


def results_dir():
    """Root directory MOABB stores results in, ``MOABB_RESULTS`` or ``~/mne_data``."""
    root = get_config("MOABB_RESULTS") or get_config("MNE_DATA")
    return Path(root) if root else Path.home() / "mne_data"


def _load_contexts(contexts):
    if contexts is None:
        return {}
    if isinstance(contexts, Mapping):
        return dict(contexts)
    with Path(contexts).open() as stream:
        return yaml.safe_load(stream) or {}


def load_pipeline_file(path):
    """Load one pipeline definition file, either YAML or Python."""
    path = Path(path)
    if path.suffix == ".yml":
        return parse_pipelines_from_directory(str(path))
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return [module.PIPELINE]


def _load_pipeline_configs(pipelines):
    """Load the named definition files, or pass through configuration dicts."""
    if pipelines is None:
        pipelines = DEFAULT_PIPELINES
    configs = []
    for item in pipelines:
        configs.extend([item] if isinstance(item, dict) else load_pipeline_file(item))
    return configs


def _seed_configs(configs, random_state):
    """Set ``random_state`` on every estimator in every pipeline that exposes it."""
    for config in configs:
        pipeline = config["pipeline"]
        keys = [
            key
            for key in pipeline.get_params()
            if key.rsplit("__", 1)[-1] == "random_state"
        ]
        pipeline.set_params(**dict.fromkeys(keys, random_state))
    return configs


def select_datasets(paradigm, include=None, exclude=None):
    """Return the datasets of a paradigm to evaluate.

    Parameters
    ----------
    paradigm : BaseFNIRS
        Paradigm the datasets must be compatible with.
    include : list of str or dataset | None
        Datasets to use, by object or exported class name. ``None`` uses the
        paradigm's default datasets.
    exclude : list of str or dataset | None
        Default datasets to leave out.
    """
    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include_datasets and exclude_datasets")
    if include is not None:
        # Dataset codes are the exported class names.
        named = [
            datasets.get_dataset(item) if isinstance(item, str) else item
            for item in include
        ]
        return [dataset for dataset in named if paradigm.is_valid(dataset)]
    codes = {item if isinstance(item, str) else item.code for item in exclude or ()}
    return [dataset for dataset in paradigm.datasets if dataset.code not in codes]


def _seed_suffix(suffix, seed):
    return "_".join(part for part in (suffix, f"seed-{seed}") if part)


def _run_evaluation(
    evaluation_class,
    paradigm,
    paradigm_name,
    datasets,
    pipelines,
    param_grid,
    evaluation_name,
    **kwargs,
):
    kwargs["suffix"] = _seed_suffix(kwargs["suffix"], kwargs["random_state"])
    if evaluation_name in SHUFFLED_EVALUATIONS:
        kwargs["cv_kwargs"] = {
            "shuffle": True,
            "random_state": kwargs["random_state"],
        }
    evaluation = evaluation_class(
        paradigm=paradigm,
        datasets=list(datasets),
        **kwargs,
    )
    frame = evaluation.process(pipelines=pipelines, param_grid=param_grid)
    frame["paradigm"] = paradigm_name
    frame["evaluation"] = evaluation_name
    return frame


def _save_analysis(results, output, plot):
    for (paradigm, evaluation), frame in results.groupby(["paradigm", "evaluation"]):
        path = Path(output) / paradigm / evaluation
        path.mkdir(parents=True, exist_ok=True)
        analyze(frame, str(path), plot=plot)


def _display_results(results):
    columns = ["paradigm", "dataset", "evaluation", "pipeline"]
    summary = results.groupby(columns, as_index=False)["score"].mean()
    print(summary.to_string(index=False))


def aggregate_seed_results(results):
    """Average matching MOABB result rows from repeated seeded runs."""
    combined = pd.concat(results, ignore_index=True)
    return combined.groupby(RESULT_KEYS, as_index=False, dropna=False).mean(
        numeric_only=True
    )


def fnirs_benchmark(
    pipelines=None,
    evaluations=None,
    paradigms=None,
    results=None,
    overwrite=False,
    output=None,
    suffix="",
    n_jobs=-1,
    plot=False,
    contexts=None,
    include_datasets=None,
    exclude_datasets=None,
    n_splits=5,
    cache_config=None,
    optuna=False,
    codecarbon_config=None,
    random_state=42,
    no_grid=False,
):
    """Run MOABB evaluations for configured fNIRS pipelines and paradigms.

    Parameters
    ----------
    no_grid : bool
        Ignore the ``param_grid`` a pipeline declares and fit it once with
        its declared parameters instead of searching.
    """
    results = results_dir() if results is None else Path(results)
    output = results / "benchmark" if output is None else Path(output)
    contexts = _load_contexts(contexts)
    configs = _seed_configs(_load_pipeline_configs(pipelines), random_state)
    pipelines_by_paradigm = generate_paradigms(configs, contexts, log)
    param_grid = {} if no_grid else generate_param_grid(configs, contexts, log)
    evaluation_names = evaluations or list(EVALUATIONS)
    paradigm_names = paradigms or list(pipelines_by_paradigm)
    unknown = set(paradigm_names) - PARADIGMS.keys()
    if unknown:
        raise ValueError(f"Unknown fNIRS paradigms: {sorted(unknown)}")
    if missing := set(paradigm_names) - pipelines_by_paradigm.keys():
        raise ValueError(f"No pipelines configured for paradigms: {sorted(missing)}")
    if unknown := set(evaluation_names) - EVALUATIONS.keys():
        raise ValueError(f"Unknown evaluations: {sorted(unknown)}")

    frames = []
    common = {
        "hdf5_path": str(results),
        "overwrite": overwrite,
        "n_jobs": n_jobs,
        "n_splits": n_splits,
        "cache_config": cache_config,
        "optuna": optuna,
        "codecarbon_config": codecarbon_config,
        "random_state": random_state,
        "suffix": suffix,
    }
    for paradigm_name in paradigm_names:
        paradigm = PARADIGMS[paradigm_name](**contexts.get(paradigm_name, {}))
        selected = select_datasets(paradigm, include_datasets, exclude_datasets)
        if not selected:
            log.warning("No datasets selected for %s", paradigm_name)
            continue
        for evaluation_name in evaluation_names:
            frames.append(
                _run_evaluation(
                    EVALUATIONS[evaluation_name],
                    paradigm,
                    paradigm_name,
                    selected,
                    pipelines_by_paradigm[paradigm_name],
                    param_grid,
                    evaluation_name,
                    **common,
                )
            )
    if not frames:
        raise ValueError("No compatible datasets selected")
    combined = pd.concat(frames, ignore_index=True)
    _save_analysis(combined, output, plot)
    _display_results(combined)
    return combined
