"""
Count parameters of every searched model setting and save to model_param_counts.json.

Example usage:
    uv run python scripts/paper/model_param_counts.py
"""

import argparse
import json
from pathlib import Path

from fnirsatlas.config import get_param_grids
from fnirsatlas.model_param_counts import generate_param_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(get_param_grids()))
    args = parser.parse_args()

    counts = generate_param_counts(args.models)
    output_file = Path(__file__).with_name("model_param_counts.json")
    output_file.write_text(json.dumps(counts, indent=2) + "\n")
    print(f"Parameter counts saved to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
