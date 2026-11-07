"""
Compute total raw recording hours per dataset and save to raw_hours.json.

Example usage:
    python scripts/paper/raw_hours.py
"""

import json
from pathlib import Path
from fnirsatlas.load_parser import load_parser
from fnirsatlas.parsers import get_available_datasets


def get_dataset_hours(dataset_name: str) -> float:
    """Return total raw recording hours for a dataset across all subjects/sessions."""
    parser = load_parser(dataset_name, download="no")
    total_sec = 0.0
    for subject_id in parser.get_subject_ids():
        for session in parser.get_sessions(subject_id):
            data = parser.load_data(subject_id, session=session)
            raw = parser._raw_from_session(data)
            total_sec += raw.times[-1]
    return total_sec / 3600


def main():
    raw_hours = {}
    for d in get_available_datasets():
        hours = get_dataset_hours(d)
        raw_hours[d] = round(hours, 1)
        print(f"{d:<20}: {hours:.1f}h")

    output_file = Path(__file__).with_name("raw_hours.json")
    with open(output_file, "w") as f:
        json.dump(raw_hours, f, indent=2)
    print(f"\nRaw hours saved to: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())
