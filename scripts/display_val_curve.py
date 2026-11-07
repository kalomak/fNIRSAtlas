#!/usr/bin/env python3
"""
Plot validation curves averaged across seeds.

Usage:
    python scripts/display_val_curve.py -d bnci -m eegnet --seeds 42 43 44 45 46 47
"""

import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fnirsatlas.config import get_results_cache_dir, get_results_dir
from fnirsatlas.logging import setup_logging

logger = logging.getLogger(__name__)


def load_curves(dataset, model, seeds):
    results_dir = get_results_cache_dir() / "val_curve" / model / dataset
    dfs = []
    loaded_seeds = []
    for seed in seeds:
        path = results_dir / f"seed_{seed}.csv"
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        dfs.append(df)
        loaded_seeds.append(seed)
    if not dfs:
        raise FileNotFoundError(f"No results found for {dataset}/{model} seeds={seeds}")
    return pd.concat(dfs, ignore_index=True), loaded_seeds


def plot_curve(df, dataset, model, fig_path, show=False):
    n_epochs = df["epoch_ix"].max() + 1
    plt.figure(figsize=(5, 5))
    sns.lineplot(data=df, x="epoch_ix", y="val_acc")
    plt.title(f"{dataset.capitalize()} ({model})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.xticks(np.arange(0, n_epochs + 1, 5))
    plt.xlim(-0.3, n_epochs + 0.3)
    plt.grid(True)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info(f"Saved figure to {fig_path}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Plot val curves averaged across seeds."
    )
    parser.add_argument("-d", "--datasets", nargs="+", required=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="eegnet",
        help="Model to use (fnirst, eegnet)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Do not show the plot after saving.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        try:
            df, loaded_seeds = load_curves(dataset, args.model, args.seeds)
        except FileNotFoundError as e:
            logger.error(e)
            continue
        seeds_str = "_".join(str(s) for s in loaded_seeds)
        fig_dir = get_results_dir() / "val_curve"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f"{dataset}_{args.model}_seeds{seeds_str}_curve.png"
        plot_curve(df, dataset, args.model, fig_path, show=not args.no_plot)


if __name__ == "__main__":
    main()
