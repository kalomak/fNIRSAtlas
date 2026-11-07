#!/usr/bin/env python3
"""Side-by-side CD diagrams of the paper models: cross-subject (a) and within-subject (b).

Example:
    uv run --extra aeon python scripts/paper/cd_diagram_cross_within.py
"""

from cd_diagram import cd_image, save_side_by_side
from fnirsatlas.config import get_results_dir

SEEDS = [0, 1, 2, 3, 4]


def main() -> int:
    images = {"a)": cd_image("cross", SEEDS), "b)": cd_image("within", SEEDS)}
    output_path = get_results_dir() / "paper" / "cd_diagram_cross_within_side_by_side.png"
    save_side_by_side(images, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
