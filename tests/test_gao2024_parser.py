import pandas as pd

from fnirsatlas.parsers.gao2024 import Gao2024Parser


def test_numeric_events_are_mapped_to_hand_labels(tmp_path):
    parser = Gao2024Parser(tmp_path)
    metadata = pd.DataFrame({"trial_type": [1, 2]})

    mapped = parser.apply_metadata(metadata)

    assert mapped["trial_type"].tolist() == ["Left", "Right"]


def test_only_historical_ball_squeezing_subjects_are_exposed(tmp_path):
    parser = Gao2024Parser(tmp_path)
    parser._get_bids_sub_ids = lambda: ["169", "170", "185"]

    assert parser.get_subject_ids() == ["sub-170", "sub-185"]
    assert parser.experiment == "BallSqueezing"
