"""
Demo script for processing and classifying any supported dataset.
"""

import logging
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GroupKFold

from fnirsatlas.load_parser import load_parser
from fnirsatlas.processed_dataset import to_dataset
from fnirsatlas.feature_extraction import WindowedFeatureExtractor
from fnirsatlas.logging import setup_logging
from fnirsatlas.utils import get_processed_data_path

setup_logging()
logger = logging.getLogger(__name__)

# Process dataset
aparser = argparse.ArgumentParser()
aparser.add_argument(
    "-d", "--dataset_name", type=str, nargs="?", default="bnci", help="Dataset name (default: bnci)"
)
aparser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing processed dataset",
)
args = aparser.parse_args()
dataset_name = args.dataset_name

processed_dataset_path = get_processed_data_path(dataset_name)
processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)

if not processed_dataset_path.exists() or args.overwrite:
    parser = load_parser(dataset_name)
    epochs_dict = parser.get_epochs()
    X, y, groups, info = to_dataset(epochs_dict)
    np.savez(
        processed_dataset_path, X=X, y=y, groups=groups, info=info
    )  # Save processed dataset
    logger.info(f"Saved processed dataset to {processed_dataset_path}")
else:
    logger.info(f"Loading processed dataset from {processed_dataset_path}")

# Load processed dataset
loaded = np.load(processed_dataset_path, allow_pickle=True)
X, y, groups, info = (
    loaded["X"],
    loaded["y"],
    loaded["groups"],
    loaded["info"],
)

# Classify
clf = Pipeline(
    [
        ("extractor", WindowedFeatureExtractor(n_windows=6, features=["mean"])),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")),
    ]
)
subject_ids = list(dict.fromkeys(groups))
sub_scores = []
for sub in subject_ids:
    sub_mask = groups == sub
    Xs, ys = X[sub_mask], y[sub_mask]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    sub_score = cross_val_score(clf, Xs, ys, cv=skf, scoring="accuracy")
    sub_scores.append(np.mean(sub_score))

print(f"Within-subject accuracy: {np.mean(sub_scores):.2f}")

gkf = GroupKFold(n_splits=len(subject_ids))  # Leave-one-subject-out
cross_scores = cross_val_score(clf, X, y, groups=groups, cv=gkf, scoring="accuracy")
print(f"Cross-subject accuracy:  {np.mean(cross_scores):.2f}")
