#!/usr/bin/env python3
"""Side-by-side CD diagrams of the paper models: 1 seed vs 5 seeds.

Example:
    uv run --extra aeon python scripts/paper/seed_cd_diagram.py -c within
"""

import argparse

from cd_diagram import SETTING_DEFAULT_STRATEGY, cd_image, save_side_by_side
from fnirsatlas.config import get_results_dir

PANELS = {"a) 1 repetition": [0], "b) 5 repetitions": [0, 1, 2, 3, 4]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--classification_setting", choices=list(SETTING_DEFAULT_STRATEGY), default="cross")
    setting = parser.parse_args().classification_setting
    images = {label: cd_image(setting, seeds) for label, seeds in PANELS.items()}
    output_path = get_results_dir() / "paper" / f"cd_diagram_{setting}_acc_seeds_1_vs_5.png"
    save_side_by_side(images, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
