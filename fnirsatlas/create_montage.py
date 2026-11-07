import numpy as np
from scipy.optimize import least_squares


def _split_fnirs_pair(channel_name: str):
    pair = channel_name.split()[0]
    source_name, detector_name = pair.split("_")
    return source_name, detector_name


def _ordered_sensor_names(channel_names):
    sources = []
    detectors = []
    for channel_name in channel_names:
        source_name, detector_name = _split_fnirs_pair(channel_name)
        if source_name not in sources:
            sources.append(source_name)
        if detector_name not in detectors:
            detectors.append(detector_name)
    return sources, detectors


def _default_tangent(channel_position):
    radial = channel_position / np.linalg.norm(channel_position)
    tangent = np.cross(radial, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.cross(radial, np.array([0.0, 1.0, 0.0]))
    return tangent / np.linalg.norm(tangent)


def _initial_sensor_guess(
    channel_names, channel_positions, sources, detectors, sensor_distance
):
    guesses = {name: [] for name in [*sources, *detectors]}
    for channel_name, channel_position in zip(channel_names, channel_positions):
        source_name, detector_name = _split_fnirs_pair(channel_name)
        tangent = _default_tangent(channel_position)
        offset = tangent * (sensor_distance / 2.0)
        guesses[source_name].append(channel_position - offset)
        guesses[detector_name].append(channel_position + offset)
    return np.concatenate(
        [np.mean(guesses[name], axis=0) for name in [*sources, *detectors]]
    )


def _sensor_residuals(
    flat_positions,
    channel_names,
    channel_positions,
    sources,
    detectors,
    sensor_distance,
):
    sensor_names = [*sources, *detectors]
    sensor_positions = flat_positions.reshape(len(sensor_names), 3)
    sensor_map = dict(zip(sensor_names, sensor_positions))
    target_radius = np.linalg.norm(channel_positions, axis=1).mean()
    residuals = []
    for channel_name, channel_position in zip(channel_names, channel_positions):
        source_name, detector_name = _split_fnirs_pair(channel_name)
        source_pos = sensor_map[source_name]
        detector_pos = sensor_map[detector_name]
        midpoint = (source_pos + detector_pos) / 2.0
        residuals.extend(30.0 * (midpoint - channel_position))
        residuals.append(5.0 * np.linalg.norm(source_pos - detector_pos) - sensor_distance)
    for sensor_position in sensor_positions:
        residuals.append(2.0 * (np.linalg.norm(sensor_position) - target_radius))
    return np.asarray(residuals)


def find_best_sensor_locations_for_channels(
    channel_names, channel_positions, sensor_distance=0.03
):
    """Infer source and detector positions from channel-center locations.

    Optimization targets:
    1. Channels ~= midpoint of source and detector.
    2. Source-detector distances ~= `sensor_distance`.
    3. Optode dist from origin ~= head radius.
    while the source-detector spacing is softly encouraged toward `sensor_distance`.
    """
    channel_positions = np.asarray(channel_positions, dtype=float)
    sources, detectors = _ordered_sensor_names(channel_names)
    initial_guess = _initial_sensor_guess(
        channel_names, channel_positions, sources, detectors, sensor_distance
    )
    result = least_squares(
        _sensor_residuals,
        initial_guess,
        args=(channel_names, channel_positions, sources, detectors, sensor_distance),
    )
    sensor_names = [*sources, *detectors]
    sensor_positions = result.x.reshape(len(sensor_names), 3)
    return dict(zip(sensor_names, sensor_positions))
