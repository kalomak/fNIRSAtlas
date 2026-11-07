import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    ParameterGrid,
    StratifiedShuffleSplit,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
import json
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from fnirsatlas.utils import get_tqdm_env_kwargs

from fnirsatlas.io.processed_dataset import (
    load_processed_dataset,
)
from fnirsatlas.config import get_processed_datasets_dir
from fnirsatlas.utils import map_values_to_sequential, metric_to_subject_scores
from fnirsatlas.format import format_benchmark_table
from fnirsatlas.stats import bootstrap_se
from fnirsatlas.feature_extraction import WindowedFeatureExtractor, StandardScalerND
from fnirsatlas.config import get_classification_tasks, get_cv_config, get_model_params
from fnirsatlas.dataset_catalog import get_available_datasets, get_suite_datasets
from fnirsatlas.model_selection import create_splitter


logger = logging.getLogger(__name__)


def f1_macro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def precision_macro_score(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def recall_macro_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


DEFAULT_SCORING = {
    "accuracy": accuracy_score,
    "kappa": cohen_kappa_score,
    "f1_macro": f1_macro_score,
    "precision": precision_macro_score,
    "recall": recall_macro_score,
}

SEARCH_STRATEGIES = {
    "cross": {"none", "nested", "single"},
    "within": {"none", "nested", "single", "block"},
}


def resolve_search_strategy(
    classification_setting, search_strategy=None, param_grid=None
):
    allowed = SEARCH_STRATEGIES.get(classification_setting, set())
    if search_strategy is None:
        if param_grid is None:
            return "none"
        return "nested" if classification_setting == "cross" else "block"
    if search_strategy not in allowed:
        raise ValueError(
            f"search_strategy={search_strategy!r} not supported for "
            f"classification_setting={classification_setting!r}"
        )
    if search_strategy != "none" and param_grid is None:
        raise ValueError(f"search_strategy={search_strategy!r} requires param_grid")
    return search_strategy


def get_within_subject_splits(y, seed, spec):
    counts = np.unique(y, return_counts=True)[1]
    n_splits = min(spec["kwargs"]["n_splits"], np.min(counts))
    if n_splits < 2:
        return None
    overrides = {"n_splits": n_splits}
    if spec["kwargs"].get("shuffle"):
        overrides["random_state"] = seed
    cv = create_splitter(spec, **overrides)
    return list(cv.split(np.zeros(len(y)), y))


def get_within_subject_data(X, y, groups, seed):
    spec = get_cv_config()["within"]
    subject_data = []
    for subj in np.unique(groups):
        mask = groups == subj
        splits = get_within_subject_splits(y[mask], seed, spec)
        if splits is None:
            logger.debug(
                "Skipping subject %s: fewer than 2 within-subject splits", subj
            )
            continue
        subject_data.append((subj, X[mask], y[mask], splits))
    logger.debug("Prepared within-subject CV data for %d subjects", len(subject_data))
    return subject_data


def evaluate_fixed_config(
    X,
    y,
    splits,
    get_clf,
    n_jobs,
    clf_kwargs,
    classification_setting,
    groups=None,
    param_grid=None,
    search_strategy="none",
    estimator_params=None,
):
    """Run one CV evaluation for one concrete classifier configuration."""
    seed_clf_kwargs = {**clf_kwargs, "random_state": clf_kwargs.get("random_state", 42)}
    logger.debug(
        "Evaluating fixed config: setting=%s strategy=%s splits=%d groups=%s params=%s",
        classification_setting,
        search_strategy,
        len(splits),
        None if groups is None else len(np.unique(groups)),
        estimator_params,
    )
    return cross_validate(
        X,
        y,
        splits,
        get_clf,
        n_jobs=n_jobs,
        clf_kwargs=seed_clf_kwargs,
        param_grid=param_grid,
        classification_setting=classification_setting,
        search_strategy=search_strategy,
        estimator_params=estimator_params,
        groups=groups,
    )


def evaluate_subject_candidates(
    X_subject,
    y_subject,
    splits,
    get_clf,
    n_jobs,
    clf_kwargs,
    candidate_params,
):
    """Evaluate all candidate parameter settings for one subject."""
    acc_scores, kappa_scores = [], []
    logger.debug(
        "Evaluating %d candidate configs for one subject", len(candidate_params)
    )
    for param_ix, params in enumerate(candidate_params):
        logger.debug("Evaluating candidate %d: %s", param_ix, params)
        results = evaluate_fixed_config(
            X_subject,
            y_subject,
            splits,
            get_clf,
            n_jobs,
            clf_kwargs,
            classification_setting="within",
            search_strategy="block",
            estimator_params=params,
        )
        acc_scores.append(results["test_accuracy"])
        kappa_scores.append(results["test_kappa"])
        logger.debug(
            "Candidate %d scores: acc=%.3f kappa=%.3f",
            param_ix,
            np.mean(results["test_accuracy"]),
            np.mean(results["test_kappa"]),
        )
    return acc_scores, kappa_scores


def select_block_params(mean_accuracy):
    """Choose one parameter index per held-out subject using LOSO accuracy."""
    if len(mean_accuracy) < 2:
        raise ValueError("within block search requires at least 2 valid subjects")
    selected = []
    mean_accuracy = np.asarray(mean_accuracy, dtype=float)
    for held_out in range(mean_accuracy.shape[0]):
        held_in = np.delete(mean_accuracy, held_out, axis=0)
        selected_ix = int(np.argmax(np.mean(held_in, axis=0)))
        logger.debug(
            "Selected candidate %d for held-out subject index %d",
            selected_ix,
            held_out,
        )
        selected.append(selected_ix)
    return selected


def classify_within_block(X, y, groups, get_clf, seed, n_jobs, clf_kwargs, param_grid):
    """Select within-subject params with LOSO blocks and cached subject CV scores."""
    subject_data = get_within_subject_data(X, y, groups, seed)
    candidate_params = [dict(params) for params in ParameterGrid(param_grid)]
    logger.debug(
        "Running within-subject block search for %d subjects and %d candidates",
        len(subject_data),
        len(candidate_params),
    )
    subjects, subject_acc, subject_kappa = [], [], []
    for subj, Xs, ys, splits in subject_data:
        logger.debug("Evaluating block-search candidates for subject %s", subj)
        acc_scores, kappa_scores = evaluate_subject_candidates(
            Xs, ys, splits, get_clf, n_jobs, clf_kwargs, candidate_params
        )
        subjects.append(subj)
        subject_acc.append(acc_scores)
        subject_kappa.append(kappa_scores)
    mean_acc = [
        [
            np.mean(candidate_split_scores)
            for candidate_split_scores in subject_candidate_scores
        ]
        for subject_candidate_scores in subject_acc
    ]
    selected_ix = select_block_params(mean_acc)
    logger.debug("Finished block search with selected candidates %s", selected_ix)
    return {
        "within_acc": [subject_acc[i][ix] for i, ix in enumerate(selected_ix)],
        "within_kappa": [subject_kappa[i][ix] for i, ix in enumerate(selected_ix)],
        "search": {
            "strategy": "block",
            "selection_metric": "accuracy",
            "subjects": [
                subj.item() if hasattr(subj, "item") else subj for subj in subjects
            ],
            "candidate_params": candidate_params,
            "candidate_accuracy": subject_acc,
            "candidate_kappa": subject_kappa,
            "selected_param_ix_by_subject": selected_ix,
            "selected_params_by_subject": [candidate_params[ix] for ix in selected_ix],
        },
    }


def get_baseline_model(model_name="lda", **kwargs):
    """Returns an unfitted instance of a baseline model."""
    if model_name.lower() == "lda":
        return Pipeline(
            [
                ("extractor", WindowedFeatureExtractor(n_windows=5, features=["mean"])),
                ("scaler", StandardScalerND()),
                ("clf", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")),
            ]
        )
    elif model_name.lower() == "logreg":
        random_state = kwargs.get("random_state", 42)
        return Pipeline(
            [
                ("extractor", WindowedFeatureExtractor()),
                ("scaler", StandardScalerND()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=100_000,
                        random_state=random_state,
                        solver="saga",
                    ),
                ),
            ]
        )
    elif model_name.lower() == "svm":
        random_state = kwargs.get("random_state", 42)
        return Pipeline(
            [
                ("extractor", WindowedFeatureExtractor()),
                ("scaler", StandardScalerND()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        gamma="scale",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_name.lower() == "rf":
        random_state = kwargs.get("random_state", 42)
        return Pipeline(
            [
                ("extractor", WindowedFeatureExtractor()),
                ("scaler", StandardScalerND()),
                (
                    "clf",
                    RandomForestClassifier(random_state=random_state, n_jobs=1),
                ),
            ]
        )
    elif model_name.lower() == "knn":
        return Pipeline(
            [
                ("extractor", WindowedFeatureExtractor()),
                ("scaler", StandardScalerND()),
                ("clf", KNeighborsClassifier(n_jobs=1)),
            ]
        )
    elif model_name.lower() == "ann":
        random_state = kwargs.get("random_state", 42)
        return Pipeline(
            [
                (
                    "extractor",
                    WindowedFeatureExtractor(n_windows=1, features=["mean", "slope"]),
                ),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(20,),
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_name.lower() == "eegnet":
        try:
            from fnirsatlas.dl.EEGNet import create_eegnet_pipeline

            default_params = get_model_params("eegnet")
            default_params.update(kwargs)
            return create_eegnet_pipeline(**default_params)
        except ImportError:
            raise ImportError(
                "EEGNet import failed. Install pytorch: pip install torch"
            )
    elif model_name.lower() == "dcn":
        try:
            from fnirsatlas.dl.DCN import create_dcn_pipeline

            default_params = get_model_params("dcn")
            default_params.update(kwargs)
            return create_dcn_pipeline(**default_params)
        except ImportError:
            raise ImportError("DCN import failed. Install pytorch: pip install torch")
    elif model_name.lower() == "lstm":
        try:
            from fnirsatlas.dl.LSTM import create_lstm_pipeline

            default_params = get_model_params("lstm")
            default_params.update(kwargs)
            return create_lstm_pipeline(**default_params)
        except ImportError:
            raise ImportError("LSTM import failed. Install pytorch: pip install torch")
    elif model_name.lower() == "fnirsnet23":
        try:
            from fnirsatlas.dl.fNIRSNet23 import create_fnirsnet23_pipeline

            default_params = get_model_params("fnirsnet23")
            default_params.update(kwargs)
            return create_fnirsnet23_pipeline(**default_params)
        except ImportError:
            raise ImportError(
                "fNIRSNet23 import failed. Install pytorch: pip install torch"
            )
    elif model_name.lower() == "fnirsnet24":
        try:
            from fnirsatlas.dl.fNIRSNet24 import create_fnirsnet24_pipeline

            default_params = get_model_params("fnirsnet24")
            default_params.update(kwargs)
            return create_fnirsnet24_pipeline(**default_params)
        except ImportError:
            raise ImportError(
                "fNIRSNet24 import failed. Install pytorch: pip install torch"
            )
    elif model_name.lower() == "cnn2a":
        try:
            from fnirsatlas.dl.CNN2a import create_cnn2a_pipeline

            default_params = get_model_params("cnn2a")
            default_params.update(kwargs)
            return create_cnn2a_pipeline(**default_params)
        except ImportError:
            raise ImportError("CNN2a import failed. Install pytorch: pip install torch")
    elif model_name.lower() == "fnirst":
        try:
            from fnirsatlas.dl.transformers import create_transformer_pipeline

            default_params = get_model_params("fnirst")
            default_params.update(kwargs)
            return create_transformer_pipeline(**default_params)
        except ImportError:
            raise ImportError(
                "Transformer import failed. Install dependencies: pip install torch einops"
            )
    elif model_name.lower() in ["tscnn", "stcnn"]:
        try:
            from fnirsatlas.dl.TSCNN import create_tscnn_pipeline, create_stcnn_pipeline

            default_params = get_model_params(model_name.lower())
            default_params.update(kwargs)
            factory = (
                create_tscnn_pipeline
                if model_name.lower() == "tscnn"
                else create_stcnn_pipeline
            )
            return factory(**default_params)
        except ImportError:
            raise ImportError(
                f"{model_name.upper()} import failed. Install pytorch: pip install torch"
            )
    elif model_name.lower().startswith("aeon_"):
        try:
            from fnirsatlas.aeon import create_aeon_pipeline

            default_params = get_model_params(model_name.lower())
            default_params.update(kwargs)
            return create_aeon_pipeline(model_name.lower(), **default_params)
        except ImportError:
            raise ImportError("Aeon import failed. Install: pip install aeon")
    else:
        raise ValueError(f"Model name {model_name} not supported")


def classify_cross(
    X,
    y,
    groups,
    get_clf=None,
    seed=42,
    n_jobs=1,
    clf_kwargs={},
    split_ix=None,
    search_strategy=None,
    param_grid=None,
):
    if get_clf is None:
        get_clf = get_baseline_model
    clf_kwargs = dict(clf_kwargs)
    if search_strategy is None:
        search_strategy = clf_kwargs.pop("search_strategy", None)
    if param_grid is None:
        param_grid = clf_kwargs.pop("param_grid", None)
    clf_kwargs.pop("inner_cv_results_path", None)
    search_strategy = resolve_search_strategy("cross", search_strategy, param_grid)
    clf_kwargs["classification_setting"] = "cross"

    # Get CV configuration
    spec = get_cv_config()["cross"]
    n_splits = min(spec["kwargs"]["n_splits"], len(np.unique(groups)))
    overrides = {"n_splits": n_splits}
    if spec["kwargs"].get("shuffle"):
        overrides["random_state"] = seed
    cv = create_splitter(spec, **overrides)
    splits = list(cv.split(X, y, groups))
    if split_ix is not None:
        if split_ix < 0 or split_ix >= len(splits):
            raise ValueError(
                f"split_ix={split_ix} is out of range for {len(splits)} splits"
            )
        splits = [splits[split_ix]]
    seed_clf_kwargs = {**clf_kwargs, "random_state": seed}
    results = cross_validate(
        X,
        y,
        splits,
        get_clf,
        n_jobs=n_jobs,
        clf_kwargs=seed_clf_kwargs,
        param_grid=param_grid,
        classification_setting="cross",
        search_strategy=search_strategy,
        groups=groups,
    )

    out = {
        "cross_acc": results["test_accuracy"],
        "cross_kappa": results["test_kappa"],
        "cross_f1_macro": results["test_f1_macro"],
        "cross_precision": results["test_precision"],
        "cross_recall": results["test_recall"],
    }
    if "search_results" in results:
        out["search_results"] = results["search_results"]
    return out


def classify_within(
    X,
    y,
    groups,
    get_clf=None,
    seed=42,
    n_jobs=1,
    clf_kwargs={},
    search_strategy=None,
    param_grid=None,
):
    if get_clf is None:
        get_clf = get_baseline_model
    clf_kwargs = dict(clf_kwargs)
    if search_strategy is None:
        search_strategy = clf_kwargs.pop("search_strategy", None)
    if param_grid is None:
        param_grid = clf_kwargs.pop("param_grid", None)
    clf_kwargs.pop("inner_cv_results_path", None)
    search_strategy = resolve_search_strategy("within", search_strategy, param_grid)
    clf_kwargs["classification_setting"] = "within"
    if search_strategy == "block":
        return classify_within_block(
            X, y, groups, get_clf, seed, n_jobs, clf_kwargs, param_grid
        )

    within_acc, within_kappa, within_search = [], [], []
    within_f1_macro, within_precision, within_recall = [], [], []
    for subj, Xs, ys, splits in get_within_subject_data(X, y, groups, seed):
        results = evaluate_fixed_config(
            Xs,
            ys,
            splits,
            get_clf,
            n_jobs,
            {**clf_kwargs, "random_state": seed},
            classification_setting="within",
            param_grid=param_grid,
            search_strategy=search_strategy,
        )
        within_acc.append(results["test_accuracy"])
        within_kappa.append(results["test_kappa"])
        within_f1_macro.append(results["test_f1_macro"])
        within_precision.append(results["test_precision"])
        within_recall.append(results["test_recall"])
        if "search_results" in results:
            within_search.extend(results["search_results"])
    out = {
        "within_acc": within_acc,
        "within_kappa": within_kappa,
        "within_f1_macro": within_f1_macro,
        "within_precision": within_precision,
        "within_recall": within_recall,
    }
    if within_search:
        out["search_results"] = within_search
    return out


def classify_task(
    dataset_name,
    task,
    X=None,
    y=None,
    groups=None,
    info=None,
    classification_setting="cross",
    n_jobs=1,
    seed=42,
    get_clf=None,
    split_ix=None,
    search_strategy=None,
    **model_kwargs,
):
    """Classify a specific task from a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to classify
    task : dict
        Task dictionary containing mapping and description
    X : np.ndarray, optional
        Pre-loaded data array. If None, data will be loaded from dataset_name
    y : np.ndarray, optional
        Pre-loaded labels. If None, data will be loaded from dataset_name
    groups : np.ndarray, optional
        Pre-loaded group assignments. If None, data will be loaded from dataset_name
    info : dict, optional
        Pre-loaded dataset information. If None, data will be loaded from dataset_name
    classification_setting : str
        Type of classification: "cross" or "within"
    n_jobs : int
        Number of jobs for parallel execution
    seed : int
        Random seed for reproducibility
    get_clf : callable, optional
        Function that returns a classifier. If None, uses get_baseline_model
    split_ix : int, optional
        Specific split index to run (0-based). If None, runs all splits.
    **model_kwargs
        Keyword arguments for model creation

    Returns
    -------
    dict
        Dictionary containing results for the task
    """
    # Load data if not provided
    if X is None or y is None or groups is None or info is None:
        processed_dir = get_processed_datasets_dir()
        X, y, groups, info = load_processed_dataset(
            processed_dir / f"{dataset_name}_processed.npz"
        )

    # Prepare task-specific data
    mapping = task["mapping"]
    mapping = map_values_to_sequential(mapping)
    label2cond = {v: k for k, v in info["cond2label"].items()}
    conditions = np.array([label2cond[yy] for yy in y])
    valid_mask = np.isin(conditions, list(mapping.keys()))
    X_task = X[valid_mask]
    groups_task = groups[valid_mask]
    y_task = np.array([mapping[c] for c in conditions[valid_mask]])

    logger.info(f"Task: {task['description']}")
    logger.info(f"Mapping: {mapping}")
    logger.info(f"Classes: {np.unique(y_task, return_counts=True)}")

    model_kwargs = dict(model_kwargs)
    param_grid = model_kwargs.pop("param_grid", None)
    model_kwargs.pop("inner_cv_results_path", None)
    resolved_search_strategy = resolve_search_strategy(
        classification_setting,
        search_strategy,
        param_grid,
    )

    clf_kwargs = {
        "X": X_task,
        "y": y_task,
        "info": info,
        "classification_setting": classification_setting,
        **model_kwargs,
    }

    results = {
        "n_classes": len(np.unique(y_task)),
        "n_epochs": X_task.shape[0],
        "mapping": mapping,
        "search_strategy": resolved_search_strategy,
    }

    if classification_setting == "cross":
        cross_res = classify_cross(
            X_task,
            y_task,
            groups_task,
            get_clf=get_clf,
            n_jobs=n_jobs,
            seed=seed,
            clf_kwargs=clf_kwargs,
            split_ix=split_ix,
            search_strategy=resolved_search_strategy,
            param_grid=param_grid,
        )
        results.update(cross_res)

    if classification_setting == "within":
        within_res = classify_within(
            X_task,
            y_task,
            groups_task,
            get_clf=get_clf,
            n_jobs=n_jobs,
            seed=seed,
            clf_kwargs=clf_kwargs,
            search_strategy=resolved_search_strategy,
            param_grid=param_grid,
        )
        results.update(within_res)

    return results


def classify_dataset(
    dataset_name,
    classification_setting="cross",
    n_jobs=1,
    seed=42,
    get_clf=None,
    split_ix=None,
    search_strategy=None,
    **model_kwargs,
):
    """Classify a dataset using specified cross-validation type.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to classify
    classification_setting : str
        Type of classification: "cross" or "within"
    n_jobs : int
        Number of jobs for parallel execution
    seed : int
        Random seed for reproducibility
    get_clf : callable, optional
        Function that returns a classifier. If None, uses get_baseline_model
    split_ix : int, optional
        Specific split index to run (0-based). If None, runs all splits.
    **model_kwargs
        Keyword arguments for model creation

    Returns
    -------
    dict
        Dictionary containing results for each task
    """
    # Load data first to get the mapping for tasks
    processed_dir = get_processed_datasets_dir()
    X, y, groups, info = load_processed_dataset(
        processed_dir / f"{dataset_name}_processed.npz"
    )

    # Get tasks using the mapping from the loaded data
    tasks = get_classification_tasks(dataset_name, cond2label=info["cond2label"])
    results = {}

    for task in tasks:
        task_results = classify_task(
            dataset_name,
            task,
            X=X,
            y=y,
            groups=groups,
            info=info,
            classification_setting=classification_setting,
            n_jobs=n_jobs,
            seed=seed,
            get_clf=get_clf,
            split_ix=split_ix,
            search_strategy=search_strategy,
            **model_kwargs,
        )
        results[task["description"]] = task_results

    return results


def print_results(results, dataset_name=""):
    logger.info(
        f"{dataset_name} Cross acc: {np.mean(metric_to_subject_scores(results['cross_acc'])):.2f}"
    )
    logger.info(
        f"{dataset_name} Cross kappa: {np.mean(metric_to_subject_scores(results['cross_kappa'])):.2f}"
    )
    logger.info(
        f"{dataset_name} Within-subject acc: {np.mean(metric_to_subject_scores(results['within_acc'])):.2f}"
    )
    logger.info(
        f"{dataset_name} Within-subject kappa: {np.mean(metric_to_subject_scores(results['within_kappa'])):.2f}"
    )


def cross_validate(
    X,
    y,
    cv,
    get_clf,
    scoring=None,
    n_jobs=1,
    clf_kwargs={},
    param_grid=None,
    classification_setting="cross",
    search_strategy="none",
    estimator_params=None,
    groups=None,
):
    """Custom cross-validation using factory functions. This makes it easier to use with torch models.

    If param_grid is provided and search_strategy is "nested", performs nested cross-validation.

    X, y: data to cross-validate on.
    cv: cross-validator. Must be an iterable of (train_idx, test_idx) pairs.
    get_clf: function that returns a classifier.
    scoring: dictionary of scoring functions.
    n_jobs: number of jobs to run in parallel.
    clf_kwargs: keyword arguments for classifier creation.
    param_grid: parameter grid for nested search.
    classification_setting: cross or within.
    search_strategy: none, nested, or the outer strategy context.
    estimator_params: fixed estimator params to apply before fitting.
    groups: group labels aligned with X/y. Used for group-aware inner CV.
    """
    if scoring is None:
        scoring = DEFAULT_SCORING.copy()

    clf_kwargs = dict(clf_kwargs)
    use_grid_search = param_grid is not None and search_strategy in {"nested", "single"}

    cv_config = get_cv_config() if use_grid_search else None
    if use_grid_search:
        if classification_setting == "cross":
            inner_spec = cv_config["cross"]["inner"]
        elif classification_setting == "within":
            inner_spec = cv_config["within"]["inner"]
        else:
            inner_spec = cv_config["cross"]["inner"]
        inner_cv_k = inner_spec["kwargs"]["n_splits"]
        scorers = {name: make_scorer(func) for name, func in scoring.items()}
        refit_metric = list(scoring.keys())[0] if len(scoring) > 1 else True

    def _score_fold(y_true, y_pred):
        return {name: func(y_true, y_pred) for name, func in scoring.items()}

    def _get_inner_cv(groups_train):
        if classification_setting != "cross":
            if search_strategy == "single":
                return StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=1 / inner_cv_k,
                    random_state=clf_kwargs.get("random_state", None),
                )
            return create_splitter(inner_spec)
        if groups_train is None:
            return None
        n_groups = len(np.unique(groups_train))
        if n_groups < 2:
            return None
        if search_strategy == "single":
            val_size = max(1, round(n_groups / inner_cv_k))
            return GroupShuffleSplit(
                n_splits=1,
                test_size=val_size,
                random_state=clf_kwargs.get("random_state", None),
            )
        return create_splitter(inner_spec, n_splits=min(inner_cv_k, n_groups))

    def _fit_with_grid_search(X_train, y_train, groups_train, fold_idx):
        base_clf = get_clf(**clf_kwargs)
        inner_cv = _get_inner_cv(groups_train)
        if classification_setting == "cross" and inner_cv is None:
            logger.warning(
                "Skipping inner CV for fold %s: fewer than 2 training groups.",
                fold_idx,
            )
            return base_clf.fit(X_train, y_train)

        gs = GridSearchCV(
            base_clf,
            param_grid,
            scoring=scorers,
            cv=inner_cv,
            n_jobs=1,
            refit=refit_metric,
            verbose=logger.getEffectiveLevel() <= logging.DEBUG,
        )
        fit_kwargs = {}
        if classification_setting == "cross":
            fit_kwargs["groups"] = groups_train
        gs.fit(X_train, y_train, **fit_kwargs)
        return gs

    def _fit_and_score(train_idx, test_idx, fold_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = None if groups is None else groups[train_idx]

        if use_grid_search:
            clf = _fit_with_grid_search(X_train, y_train, groups_train, fold_idx)
            search = {"fold": fold_idx, **clf.cv_results_}
        else:
            clf = get_clf(**clf_kwargs)
            if estimator_params is not None:
                clf.set_params(**estimator_params)
            clf.fit(X_train, y_train)
            search = None
        y_pred = clf.predict(X_test)
        return _score_fold(y_test, y_pred), search

    use_tqdm = logger.isEnabledFor(logging.INFO) and classification_setting == "cross"
    cv_iter = (
        tqdm(list(enumerate(cv)), desc="CV folds") if use_tqdm else list(enumerate(cv))
    )
    if n_jobs == 1:
        results = [
            _fit_and_score(train_idx, test_idx, fold_idx)
            for fold_idx, (train_idx, test_idx) in cv_iter
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_score)(train_idx, test_idx, fold_idx)
            for fold_idx, (train_idx, test_idx) in cv_iter
        )

    scores = [r[0] for r in results]
    search_results = [r[1] for r in results if r[1] is not None]
    out = {f"test_{name}": [r[name] for r in scores] for name in scoring.keys()}
    if search_results:
        out["search_results"] = search_results
    return out


def run_benchmark(
    suite=None,
    datasets=None,
    get_clf=None,
    n_jobs=1,
    seed=42,
    classification_setting="cross",
    search_strategy=None,
    show_table=True,
    split_ix=None,
    **model_kwargs,
):
    """
    Run classification benchmark on specified datasets.

    Args:
        suite (str, optional): Suite name to classify (e.g. motor, MI, MA, etc.)
        datasets (list, optional): Specific datasets to classify
        get_clf (callable): Function that returns a classifier.
        n_jobs (int): Number of jobs for parallel execution, default=1
        seed (int): Random seed for reproducibility, default=42
        classification_setting (str): Type of classification: "cross" or "within"
        show_table (bool): Whether to show the benchmark table, default=True
        split_ix (int, optional): Specific split index to run (0-based). If None, runs all splits.
        **model_kwargs: Keyword arguments for model creation

    Returns:
        dict: Classification results for all datasets
    """
    # Get canonical dataset list from catalog
    available = get_available_datasets()

    dataset_set = set()
    if suite and suite != "all":
        dataset_set.update(get_suite_datasets(suite))
    elif datasets:
        dataset_set.update(datasets)
    else:
        dataset_set = set(available)

    # Filter to only available datasets
    dataset_set = [d for d in available if d in dataset_set]
    if not dataset_set:
        logger.error("No datasets to process")
        return {}

    all_results = {}

    # Use specific dataset name if only one dataset
    single_dataset = len(dataset_set) == 1
    desc = (
        f"{'Classifying':11} {list(dataset_set)[0][:12]:<12}"
        if single_dataset
        else "Classifying datasets"
    )

    with tqdm(dataset_set, desc=desc, **get_tqdm_env_kwargs()) as pbar:
        for name in pbar:
            if not single_dataset:
                pbar.set_postfix_str(name)
            try:
                results = classify_dataset(
                    name,
                    classification_setting=classification_setting,
                    n_jobs=n_jobs,
                    seed=seed,
                    get_clf=get_clf,
                    split_ix=split_ix,
                    search_strategy=search_strategy,
                    **model_kwargs,
                )
                all_results[name] = results

                # Log results
                for task_desc, task_results in results.items():
                    logger.info(f"Dataset: {name} | Task: {task_desc}")
                    logger.info(f"Class mapping: {task_results['mapping']}")
                    for metric in ["cross_acc", "within_acc"]:
                        if metric in task_results:
                            values = metric_to_subject_scores(task_results[metric])
                            logger.info(
                                f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f} (STD)"
                            )

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                continue

    if show_table:
        format_benchmark_table(all_results, classification_setting)
    return all_results
