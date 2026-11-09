"""fNIRS extensions of the MOABB dataset metadata schema."""

from dataclasses import dataclass

from moabb.datasets.metadata.schema import AcquisitionMetadata


@dataclass
class FnirsAcquisitionMetadata(AcquisitionMetadata):
    """fNIRS acquisition parameters.

    Extends :class:`moabb.datasets.metadata.schema.AcquisitionMetadata`.
    ``channel_types`` counts MNE fNIRS channels, one per pair and wavelength, e.g.
    ``{"fnirs_cw_amplitude": 56}``.

    Parameters
    ----------
    wavelengths_nm : List[float], optional
        Device source wavelengths in nm, ascending. None if unknown.
    n_sources : int, optional
        Number of source optodes.
    n_detectors : int, optional
        Number of detector optodes.
    n_short_channels : int, optional
        Number of short-separation source-detector pairs, 0 if none.
    sensor_positions : str, optional
        Optode position availability: ``"3d"``, ``"2d"`` or ``"none"``.
    other_modalities : List[str], optional
        Non-fNIRS modalities in the source data, e.g. ``["EEG"]``. Not loaded.
        Empty if fNIRS only.
    """

    wavelengths_nm: list[float] | None = None
    n_sources: int | None = None
    n_detectors: int | None = None
    n_short_channels: int | None = None
    sensor_positions: str | None = None
    other_modalities: list[str] | None = None
