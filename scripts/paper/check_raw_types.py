"""
Processes and saves datasets.
"""

import json
from pathlib import Path
import numpy as np
from fnirsatlas.load_parser import load_parser
from fnirsatlas.parsers import get_available_datasets


def get_raw_data_type(dataset_name: str) -> str:
    """Process and save a single dataset."""
    parser = load_parser(dataset_name, download="no")
    subject_id = parser.get_subject_ids()[0]
    session = parser.get_sessions(subject_id)[0]
    data = parser.load_data(subject_id, session=session)
    raw = parser._raw_from_session(data)
    ct = np.unique(raw.get_channel_types())
    match ct[0]:
        case "hbo" | "hbr":
            return "HbO/HbR"
        case "fnirs_od":
            return "OD"
        case "fnirs_cw_amplitude":
            return "CW"
        case _:
            raise ValueError(f"Unrecognized data type {ct[0]} for {dataset_name}")


def main():
    """Process and save a single dataset specified by command line arguments."""
    raw_types = {}
    for d in get_available_datasets():
        dt = get_raw_data_type(d)
        raw_types[d] = dt
        print(f"{d:<20}: {dt}")
    
    # Save to JSON file
    output_file = Path("scripts/paper/raw_types.json")
    with open(output_file, 'w') as f:
        json.dump(raw_types, f, indent=2)
    print(f"\nRaw types saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
