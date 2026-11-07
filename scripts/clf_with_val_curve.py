from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from fnirsatlas.classification import get_baseline_model
from fnirsatlas.feature_extraction import ChannelSplitter, StandardScalerND
from fnirsatlas.io import load_processed_dataset
from fnirsatlas.config import get_classification_tasks
from fnirsatlas.logging import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="bnci")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="fnirst",
        help="Model to use (fnirst, eegnet)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Show the plot after saving."
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.model
    seed = args.seed

    from fnirsatlas.config import get_results_cache_dir

    results_path = (
        get_results_cache_dir()
        / "val_curve"
        / model_name
        / dataset_name
        / f"seed_{seed}.csv"
    )
    if results_path.exists() and not args.override:
        logger.info(
            f"Output file already exists: {results_path} (use --override to overwrite)"
        )
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y, groups, info = load_processed_dataset(dataset_name)
    logger.info(
        f"Loaded dataset {dataset_name} with shape {X.shape} and {len(np.unique(y))} classes."
    )
    task = get_classification_tasks(dataset_name, info["cond2label"])[0]
    mapping = task["mapping"]
    label2cond = {v: k for k, v in info["cond2label"].items()}
    conditions = np.array([label2cond[yy] for yy in y])
    keep_mask = np.isin(conditions, list(mapping.keys()))
    y = np.array([mapping[c] for c in conditions[keep_mask]])
    X = X[keep_mask]
    groups = groups[keep_mask]
    X = ChannelSplitter(info["ch_names"]).fit_transform(X)

    _, _, n_chs, T = X.shape
    num_classes = np.unique(y).shape[0]
    n_epochs = 50

    split_metrics = defaultdict(list)
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_ix, test_ix in tqdm(
        list(cv.split(X, y, groups)),
        desc=f"{dataset_name} 5-fold CV (seed {seed})",
    ):
        X_train, y_train = X[train_ix], y[train_ix]
        X_test, y_test = X[test_ix], y[test_ix]
        scaler = StandardScalerND()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pipeline = get_baseline_model(
            model_name,
            X=X_train,
            y=y_train,
            info=info,
            n_class=num_classes,
            sampling_point=T,
            n_epochs=n_epochs,
        )
        model = pipeline.named_steps["clf"]
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        for k, v in model.val_metrics.items():
            split_metrics[k].append(v)
        split_metrics["test_ix"].append(test_ix)

    y_true = np.concatenate([y_true[-1] for y_true in split_metrics["y_true"]])
    y_pred = np.concatenate([y_pred[-1] for y_pred in split_metrics["y_pred"]])
    acc = accuracy_score(y_true, y_pred.argmax(axis=-1))
    kappa_score = cohen_kappa_score(y_true, y_pred.argmax(axis=-1))
    print(f"5-fold CV acc: {acc:.3f}, kappa: {kappa_score:.3f}")

    rows = []
    n_splits = len(split_metrics["val_acc"])
    n_epochs = len(split_metrics["val_acc"][0])
    for split_ix in range(n_splits):
        for epoch_ix in range(n_epochs):
            rows.append(
                {
                    "split_ix": split_ix,
                    "epoch_ix": epoch_ix,
                    "val_acc": split_metrics["val_acc"][split_ix][epoch_ix],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    if args.plot:
        plt.figure(figsize=(5, 5))
        sns.lineplot(data=df, x="epoch_ix", y="val_acc")
        plt.title(f"{dataset_name.capitalize()} 5-fold CV")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
        plt.xticks(np.arange(0, n_epochs + 1, 5))
        plt.xlim(0 - 0.3, n_epochs + 0.3)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
