import mne
import numpy as np

from fnirsatlas.parsers.base import BaseFNIRSParser

# Two runs of Gao2024 sub-179 whose sfreq differ only by float rounding.
RUN_SFREQS = [11.62574404761905, 11.625744047619047]


class TwoRunParser(BaseFNIRSParser):
    dataset_id = "gao2024"

    def get_subject_ids(self):
        return ["sub-01"]

    def get_sessions(self, subject_id):
        return [0, 1]

    def load_data(self, subject_id, session=None):
        return {"sfreq": RUN_SFREQS[session]}

    def _raw_from_session(self, data):
        sfreq = data["sfreq"]
        info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], sfreq, "hbo")
        info.set_channel_types({"S1_D1 hbr": "hbr"})
        raw = mne.io.RawArray(np.random.randn(2, int(120 * sfreq)), info, verbose=False)
        raw.set_annotations(mne.Annotations([10.0, 40.0, 70.0], 0.0, ["A", "B", "A"]))
        return raw

    def _process_raw(self, raw):
        return raw


def test_runs_with_rounding_different_sfreq_concatenate(tmp_path):
    epochs = TwoRunParser(tmp_path).get_epochs(disable_tqdm=True)["sub-01"]

    assert len(epochs) == 6
    assert epochs.info["sfreq"] == 5.0
