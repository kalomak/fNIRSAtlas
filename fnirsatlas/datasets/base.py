"""Shared base helpers for fNIRS datasets."""

import json
from typing import ClassVar

import mne
import numpy as np
from mne.io.constants import FIFF
from moabb.datasets.base import (
    BaseDataset,
    MetaclassDataset,
    _metadata_admonition_block,
)
from moabb.datasets.metadata.schema import DatasetMetadata
from moabb.datasets.preprocessing import FixedPipeline

from .utils import standardize_raw, warn_if_optodes_are_planar


class MetaclassFNIRSDataset(MetaclassDataset):
    """Adds an fNIRS block after MOABB's metadata blocks in the class docstring."""

    def __new__(cls, name, bases, attrs):
        doc = attrs.get("__doc__") or ""
        metadata = attrs.get("METADATA")
        if metadata is not None and doc.strip():
            acquisition = metadata.acquisition
            block = _metadata_admonition_block(
                "fNIRS",
                [
                    ("Wavelengths (nm)", acquisition.wavelengths_nm),
                    ("Sources", acquisition.n_sources),
                    ("Detectors", acquisition.n_detectors),
                    ("Short channels", acquisition.n_short_channels),
                    ("Sensor positions", acquisition.sensor_positions),
                    ("Other modalities", acquisition.other_modalities),
                ],
                doc,
            )
            # None when all fields are unset.
            if block is not None:
                # MOABB inserts its blocks after the first paragraph, before this
                # one.
                first, *rest = doc.split("\n\n")
                attrs["__doc__"] = "\n\n".join([first, block, *rest])
        return super().__new__(cls, name, bases, attrs)


class BaseFNIRSDataset(BaseDataset, metaclass=MetaclassFNIRSDataset):
    """Base dataset with shared fNIRS preprocessing."""

    # Preprocessing options this dataset replaces, e.g. a device-specific
    # short_channel_threshold. See fnirsatlas.paradigms.fnirs.PREPROCESSING.
    preprocessing_options: ClassVar = {}
    # Optional mapping from source annotation labels to canonical event names.
    _annotation_rename_map = None
    _FNIRS_COIL_TYPES: ClassVar = {
        "fnirs_cw_amplitude": FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE,
        "fnirs_od": FIFF.FIFFV_COIL_FNIRS_OD,
        "hbo": FIFF.FIFFV_COIL_FNIRS_HBO,
        "hbr": FIFF.FIFFV_COIL_FNIRS_HBR,
    }

    @property
    def metadata(self) -> DatasetMetadata | None:
        return getattr(type(self), "METADATA", None)

    def _create_process_pipeline(self):
        if self.interval is None:
            return FixedPipeline([])
        return super()._create_process_pipeline()

    def _standardize_raw(self, raw):
        raw = standardize_raw(raw, self.event_id, self._annotation_rename_map)
        # Checked here, where the loader has had its chance to set a montage.
        warn_if_optodes_are_planar(raw)
        return self._after_preprocess(raw)

    def _after_preprocess(self, raw):
        """Apply dataset-specific changes after fNIRS preprocessing."""
        return raw

    @staticmethod
    def _make_stim_raw(n_times, sfreq, samples, labels, channel_name="Stim"):
        samples = list(samples)
        labels = list(labels)
        if len(samples) != len(labels):
            raise ValueError("samples and labels must have the same length")
        stim = mne.io.RawArray(
            data=[[0.0] * n_times],
            info=mne.create_info([channel_name], float(sfreq), ["stim"]),
            verbose=False,
        )
        stim._data[0, samples] = labels
        return stim

    @staticmethod
    def _load_montage_from_json(path):
        """Load a montage-toolkit export.

        Parameters
        ----------
        path : Path
            JSON file with ``montage`` (``make_dig_montage`` arguments) and
            optionally ``channel_map`` (native channel identifier to
            ``S<n>_D<n>``).

        Returns
        -------
        montage : mne.channels.DigMontage
        channel_map : dict of str to str, or None if the file has none
        """
        with open(path) as handle:
            export = json.load(handle)
        montage = mne.channels.make_dig_montage(**export["montage"])
        return montage, export.get("channel_map")

    @staticmethod
    def _make_head_montage(ch_pos, fiducials):
        return mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame="head",
            **fiducials,
        )

    @staticmethod
    def _make_raw(data, ch_names, ch_types, sfreq, montage=None):
        raw = mne.io.RawArray(
            np.asarray(data),
            mne.create_info(ch_names, float(sfreq), ch_types),
            verbose=False,
        )
        if montage is not None:
            raw.set_montage(montage)
        return raw

    @staticmethod
    def _single_run_session(raw, session_key="0fnirs", run_key="0"):
        return {session_key: {run_key: raw}}

    @staticmethod
    def _add_run(sessions, session_key, raw, run_key="0"):
        sessions.setdefault(session_key, {})[run_key] = raw
        return sessions

    @classmethod
    def _populate_fnirs_metadata(cls, raw, default_wavelengths=None):
        for ch in raw.info["chs"]:
            ch_type = raw.get_channel_types(picks=[ch["ch_name"]])[0]
            coil_type = cls._FNIRS_COIL_TYPES.get(ch_type)
            if coil_type is None:
                continue
            ch["kind"] = FIFF.FIFFV_FNIRS_CH
            ch["coil_type"] = coil_type
            ch["unit"] = FIFF.FIFF_UNIT_MOL
            if np.isnan(ch["loc"][9]):
                wavelength = cls._get_channel_wavelength(
                    ch["ch_name"], ch_type, default_wavelengths
                )
                if wavelength is not None:
                    ch["loc"][9] = wavelength
        return raw

    @staticmethod
    def _get_channel_wavelength(ch_name, ch_type, default_wavelengths=None):
        if ch_type in {"fnirs_cw_amplitude", "fnirs_od"}:
            return float(ch_name.split()[-1])
        if default_wavelengths is None:
            return None
        if ch_type == "hbo":
            return float(default_wavelengths[0])
        if ch_type == "hbr":
            return float(default_wavelengths[1])
        return None
