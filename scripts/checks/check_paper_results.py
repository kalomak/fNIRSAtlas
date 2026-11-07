"""
Check which paper results are present across all datasets, models, and seeds.

Usage:
    python scripts/checks/check_paper_results.py
    python scripts/checks/check_paper_results.py --models lda eegnet --seeds 0 1 2
"""

import argparse
from fnirsatlas.config import get_paper_models, get_results_cache_dir
from fnirsatlas.parsers import get_available_datasets

N_SPLITS = 5

MODELS = get_paper_models()
SEEDS = range(5)
SEARCH_STRATEGIES = {"cross": "nested", "within": "block"}


def result_present(results_dir, dataset, setting, model, seed):
    """Check for either an aggregated result or all split results."""
    strategy = SEARCH_STRATEGIES[setting]
    seed_dir = results_dir / setting / strategy / model / dataset / str(seed)
    if (seed_dir / "results.json").exists():
        return True
    return all(
        (seed_dir / f"results_split{i}.json").exists()
        for i in range(N_SPLITS)
    )


def seeds_cell(hits, seeds):
    """Compact cell: '0,42' for present seeds, or 'all'/'--'."""
    present = [s for s, ok in zip(seeds, hits) if ok]
    if not present:
        return "--"
    if len(present) == len(seeds):
        return "all"
    return ",".join(str(s) for s in present)


def check_results(datasets, models, seeds, setting_filter=None):
    settings = ["cross", "within"]
    if setting_filter:
        settings = [s for s in settings if s == setting_filter]

    results_dir = get_results_cache_dir()
    present = {setting: 0 for setting in settings}
    col_w = max(len(m) for m in models) + 2
    ds_w = max(len(d) for d in datasets) + 2

    for setting in settings:
        print(f"\n--- {setting} ---")
        header = f"{'dataset':<{ds_w}}" + "".join(f"{m:^{col_w}}" for m in models)
        print(header)
        print("-" * len(header))
        for ds in datasets:
            row = f"{ds:<{ds_w}}"
            for model in models:
                hits = [
                    result_present(results_dir, ds, setting, model, s) for s in seeds
                ]
                present[setting] += sum(hits)
                row += f"{seeds_cell(hits, seeds):^{col_w}}"
            print(row)

    print("\n\nSummary:")
    for setting in settings:
        total = len(datasets) * len(models) * len(seeds)
        print(f"  {setting:<20}: {present[setting]}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Check paper results presence")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("-c", "--setting", choices=["cross", "within"], default=None)
    args = parser.parse_args()

    datasets = args.datasets or get_available_datasets()
    check_results(datasets, args.models, args.seeds, args.setting)


if __name__ == "__main__":
    main()
