"""Approximate Shin2018b sensor locations from channel-center coordinates.

The public Shin2018b montage only contains channel locations, not explicit source
and detector positions. This script was used to infer optode locations from the
paper's 5-source / 3-detector layout, visualize the result against standard head
positions, and export the inferred montage to `fnirsatlas/montages/shin2018b_montage_3d.json`.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
from fnirsatlas.montage import montage_to_dict
from fnirsatlas.config import get_results_dir
from scipy.io import loadmat

from fnirsatlas.create_montage import find_best_sensor_locations_for_channels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "~/mne_data/fnirsatlas/datasets/Shin2018b/matdata/NIRS"
        ).expanduser(),
    )
    parser.add_argument("--subject", default="VP001")
    parser.add_argument(
        "--output",
        type=Path,
        default=get_results_dir() / "checks/shin2018b_sensors.png",
    )
    parser.add_argument(
        "--montage-output",
        type=Path,
        default=Path("fnirsatlas/montages/shin2018b_montage_3d.json"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


SCALE = 0.08
COORD_FRAME = "head"
NASION = np.array([0.00146763, 0.08506715, -0.03483611])
LPA = np.array([-0.08061612, -0.02908875, -0.04131077])
RPA = np.array([0.08436285, -0.02850276, -0.04127743])


def _load_subject(subject_dir: Path):
    cnt = loadmat(subject_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)[
        "cnt"
    ]
    mnt = loadmat(subject_dir / "mnt.mat", squeeze_me=True, struct_as_record=False)[
        "mnt"
    ]
    cnt_labels = [str(x) for x in np.atleast_1d(cnt.oxy.clab).tolist()]
    mnt_labels = [str(x) for x in np.atleast_1d(mnt.clab).tolist()]
    positions = np.asarray(mnt.pos_3d).T
    return cnt_labels, mnt_labels, positions


def _extract_standard_labels(labels: list[str], candidates: set[str]) -> list[str]:
    selected = set()
    ordered = sorted(candidates, key=len, reverse=True)
    for label in labels:
        remainder = label
        parts = []
        while remainder:
            match = next((name for name in ordered if remainder.startswith(name)), None)
            if match is None:
                break
            parts.append(match)
            remainder = remainder[len(match) :]
        if "".join(parts) == label:
            selected.update(parts)
        elif label in candidates:
            selected.add(label)
    return sorted(selected)


def _match_head_scale(reference_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    ref_scale = np.linalg.norm(reference_pos, axis=1).mean()
    target_scale = np.linalg.norm(target_pos, axis=1).mean()
    return reference_pos * (target_scale / ref_scale)


def _plot_head_sphere(ax, radius: float):
    u, v = np.mgrid[np.pi / 4 : 3 * np.pi / 4 : 24j, 0 : np.pi / 2 : 16j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color="0.75", linewidth=0.5, alpha=0.35)


def _split_cnt_label(label: str) -> tuple[str, str]:
    return label[:3], label[-3:]


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _build_channel_mapping(cnt_labels: list[str]):
    source_mne_names = _ordered_unique(
        [_split_cnt_label(label)[0] for label in cnt_labels]
    )
    detector_mne_names = _ordered_unique(
        [_split_cnt_label(label)[1] for label in cnt_labels]
    )
    source_map = {name: f"S{i}" for i, name in enumerate(source_mne_names, start=1)}
    detector_map = {name: f"D{i}" for i, name in enumerate(detector_mne_names, start=1)}
    repo_labels = []
    for label in cnt_labels:
        source_name, detector_name = _split_cnt_label(label)
        repo_labels.append(f"{source_map[source_name]}_{detector_map[detector_name]}")
    return repo_labels, source_map, detector_map


def _plot_labeled_points(ax, positions, labels, color, size, zorder, vertical_offset):
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=size,
        c=color,
        zorder=zorder,
    )
    va = "bottom" if vertical_offset > 0 else "top"
    for text, (x, y, z) in zip(labels, positions):
        ax.text(
            x,
            y,
            z + vertical_offset,
            text,
            color=color,
            ha="center",
            va=va,
            fontsize=9 if color != "tab:red" else 8,
        )


def _report_mappings(
    repo_labels, source_map, detector_map, cnt_labels, mnt_labels, referenced
):
    print("dataset cnt labels:")
    print(cnt_labels)
    print("repo fnirs labels:")
    print(repo_labels)
    print("source map:")
    print(source_map)
    print("detector map:")
    print(detector_map)
    print("dataset mnt labels:")
    print(mnt_labels)
    print("referenced standard_1020 labels:")
    print(referenced)


def _sensor_positions(repo_labels, positions, source_map, detector_map):
    sensor_items = [*source_map.items(), *detector_map.items()]
    sensor_lookup = find_best_sensor_locations_for_channels(repo_labels, positions)
    sensor_pos = np.array([sensor_lookup[label] for _, label in sensor_items])
    sensor_text = [f"{label}\n({name})" for name, label in sensor_items]
    return sensor_pos, sensor_text


def _make_inferred_montage(sensor_pos, source_map, detector_map):
    sensor_items = [*source_map.items(), *detector_map.items()]
    ch_pos = {
        label: position * SCALE
        for (_, label), position in zip(sensor_items, sensor_pos)
    }
    return mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame=COORD_FRAME,
        nasion=NASION,
        lpa=LPA,
        rpa=RPA,
    )


def _save_montage(montage, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(montage_to_dict(montage), f, indent=2)
        f.write("\n")


def main():
    """Visualize and export inferred Shin2018b source and detector positions."""
    args = parse_args()
    subject_dir = args.dataset_root / args.subject
    cnt_labels, mnt_labels, positions = _load_subject(subject_dir)
    repo_labels, source_map, detector_map = _build_channel_mapping(cnt_labels)

    montage = mne.channels.make_standard_montage("standard_1020")
    std_pos = montage.get_positions()["ch_pos"]
    candidate_names = set(std_pos)
    referenced = _extract_standard_labels(cnt_labels + mnt_labels, candidate_names)

    standard_pos = np.array([std_pos[name] for name in referenced])
    standard_pos = _match_head_scale(standard_pos, positions)
    sensor_pos, sensor_text = _sensor_positions(
        repo_labels, positions, source_map, detector_map
    )
    inferred_montage = _make_inferred_montage(sensor_pos, source_map, detector_map)
    head_radius = np.linalg.norm(positions, axis=1).mean()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _plot_head_sphere(ax, head_radius)

    _plot_labeled_points(ax, standard_pos, referenced, "tab:blue", 55, 2, 0.02)
    ax.collections[-1].set_label("standard_1020 refs")
    _plot_labeled_points(ax, sensor_pos, sensor_text, "black", 35, 4, 0.01)
    ax.collections[-1].set_label("fnirs sensors")
    channel_text = [
        f"{repo_label}\n{cnt_label}\n({mnt_label})"
        for repo_label, cnt_label, mnt_label in zip(repo_labels, cnt_labels, mnt_labels)
    ]
    _plot_labeled_points(ax, positions, channel_text, "tab:red", 65, 5, -0.02)
    ax.collections[-1].set_label("Shin2018b mnt.pos_3d")

    ax.set_title(f"Shin2018b vs standard_1020 ({args.subject})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.2)
    all_xyz = np.vstack([positions, standard_pos, sensor_pos])
    pad = 0.08
    ax.set_xlim(all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
    ax.set_ylim(all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
    ax.set_zlim(all_xyz[:, 2].min() - pad, all_xyz[:, 2].max() + pad)
    ax.set_box_aspect(
        (
            np.ptp(all_xyz[:, 0]) + 2 * pad,
            np.ptp(all_xyz[:, 1]) + 2 * pad,
            np.ptp(all_xyz[:, 2]) + 2 * pad,
        )
    )
    ax.view_init(elev=20, azim=-60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    _save_montage(inferred_montage, args.montage_output)
    if args.show:
        plt.show()
    print(f"saved {args.output}")
    print(f"saved {args.montage_output}")
    _report_mappings(
        repo_labels, source_map, detector_map, cnt_labels, mnt_labels, referenced
    )


if __name__ == "__main__":
    main()
