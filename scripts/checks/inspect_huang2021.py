# %%
"""
This script reads the raw data and compares it to the preprocessed data successfully.

But, only 9 out of the 68 subjects include a raw fnirs_train.csv file meaning that for the rest of the subjects most of the data is missing.
"""


#!%load_ext autoreload
#!%autoreload 2

import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne_bids
import logging
import re as re
from scipy.io import loadmat
import numpy as np


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# %%

# wls: (690, 830)
# source-detector-distances: 25mm, 35mm
# 5.2 Hz

# At each timestep there are D = 8 numerical measurements, combining 2 spatial locations, 2 concentrations (oxygenated and de-oxygenated hemoglobin), and 2 optical data types (intensity and phase).

# We interpret the channel names as follows:
# <OpticalType>–<SourceID><λIndex?><DetectorID>
# AC‑A1	    AC–A–1–1	Intensity from source A to detector 1 @ 690 nm
# AC‑A2	    AC–A–1–2	Intensity from source A to detector 2 @ 690 nm
# AC‑A21	AC–A–2–1	Intensity from source A to detector 1 @ 830 nm
# ...
# AC-C3	    AC–C–1-3    Intensity from source C to detector 3 @ 690 nm
# ...
# PH-D24	PH–D–2-4	Phase from source D to detector 4 @ 830 nm

# For this work an absolute absorption coefficient of µa = 0.01 1/mm and an absolute reduced scattering coefficient of µ0s = 1 1/mm were assumed for each wavelength.


# Which sensor does each letter correspond to?
# Not in paper, not in datasheet
# We assume A,B,C,D left to right
s2mne = {
    "A": "S1",
    "B": "S2",
    "C": "S3",
    "D": "S4",
}
d2mne = {
    "1": "D1",
    "2": "D2",
    "3": "D3",
    "4": "D4",
}
wl2mne = {
    # "1": 690,
    # "2": 830,
    # The data matches better with the following:
    "1": 830,
    "2": 690,
}


def parse_ch_name(ch):
    # <OpticalType>–<SourceID><λIndex?><DetectorID>
    m = re.match(r"^(AC|PH)-(A|B|C|D)(\d)?(\d)", ch)
    ch_type, source_id, wl_index, detector_id = m.groups()
    wl_index = wl_index or "1"
    assert int(wl_index) == 1 or int(wl_index) == 2
    assert int(detector_id) <= 4
    assert source_id in ["A", "B", "C", "D"]
    assert ch_type in ["AC", "PH"]
    ch_name = f"{s2mne[source_id]}_{d2mne[detector_id]} {wl2mne[wl_index]} {ch_type}"
    return ch_name


# A > S1, B > S2, C > S3, D > S4
# S1 D1 D2 S2 S3 D3 D4 S4


def get_montage():
    x_start = -0.075
    y = 0.075

    # # Original sensors:
    # dists = [0, 0.025, 0.01, 0.025, 0.04, 0.025, 0.01, 0.025]
    # sensors = ["S1", "D1", "D2", "S2", "S3", "D3", "D4", "S4"]

    # # Average chs (after dual slope)
    dists = [0.015, 0.03, 0.06, 0.03]
    sensors = ["S1", "D1", "S2", "D2"]
    x = x_start + np.cumsum(dists)
    sensor_positions = {s: np.array([x[i], y, 0]) for i, s in enumerate(sensors)}

    coord_frame = "head"
    nasion = np.array([0.00146763, 0.08506715, -0.03483611])
    lpa = np.array([-0.08061612, -0.02908875, -0.04131077])
    rpa = np.array([0.08436285, -0.02850276, -0.04127743])
    return mne.channels.make_dig_montage(
        ch_pos=sensor_positions,
        coord_frame=coord_frame,
        nasion=nasion,
        lpa=lpa,
        rpa=rpa,
    )


### Dual-slope method
# https://pmc.ncbi.nlm.nih.gov/articles/PMC7519873/pdf/RSINAK-000091-093702_1.pdf, Section 2.A
# SD: Single-distance measurement, uses one source and detector
# SS: Single-slope measurement, uses one source and two detectors
# DS: Dual-slope measurement:
# A DS measurement is achieved by averaging two
# SSs under the following requirements37 (the following description
# assumes each SS is made of one source and two detectors for simplicity):
# (1) a DS se t is made of two SSs (SS1 and SS2);
# (2) the difference between the source–detector distances for both SSs must be
# the same (∣ρSD2 − ρSD1∣_SS1 = ∣ρSD2 − ρSD1∣_SS2); and
# (3) the detector for the shorter SD measurement in SS1 must be the same detector for the longer SD measurement in SS2.


def get_sd_dist(src, det):
    return 0.025 if src == det.replace("D", "S") else 0.035


def deconstruct_ch_name(ch_name):
    sd, wl, ct = ch_name.split(" ")
    s, d = sd.split("_")
    return s, d, wl, ct


def construct_ch_name(s, d, wl, ct):
    return f"{s}_{d} {wl} {ct}"


def apply_dualslope_to_pair(pair_df, mu_a=0.01, mu_sp=1.0):
    ch_names = pair_df.columns
    chs = np.array([deconstruct_ch_name(c) for c in ch_names])
    assert pair_df.shape[1] == 4
    assert chs[0][0] == chs[1][0] and chs[2][0] == chs[3][0], "Source IDs must match"
    assert chs[0][0] != chs[2][0], "Source and detector IDs must be different"
    assert chs[0][1] == chs[2][1] and chs[1][1] == chs[3][1], "Detector IDs must match"
    assert (chs[:, 2] == chs[0, 2]).all(), "Wavelengths must match"
    wl = chs[0, 2]
    assert (chs[:, 3] == chs[0, 3]).all(), "Channel types must match"
    ct = chs[0, 3]
    sd_dists = np.array([get_sd_dist(c[0], c[1]) for c in chs])
    assert (
        sd_dists[1] - sd_dists[0] == sd_dists[2] - sd_dists[3]
    ), "SD distances must be equal"
    # source, detector, wavelength, channel type, sd_dist, channel index
    chs = np.c_[chs, sd_dists, range(len(chs))]
    data_raw = pair_df.values
    if ct == "AC":
        sd = np.log(sd_dists[None, :] ** 2 * data_raw)
    elif ct == "PH":
        sd = data_raw
    ss1 = (sd[:, 1] - sd[:, 0]) / (float(chs[1, 4]) - float(chs[0, 4]))
    ss2 = (sd[:, 2] - sd[:, 3]) / (float(chs[2, 4]) - float(chs[3, 4]))
    ds = 0.5 * (ss1 + ss2)
    ds_bl = ds[:100].mean()  # baseline
    ds_delta = ds - ds_bl
    dsf = 0.5 * np.sqrt(3 * mu_sp / mu_a)
    delta_mu_a = -ds_delta / dsf
    return delta_mu_a


def apply_dualslope(df, mu_a=0.01, mu_sp=1.0):
    """
    Apply steps 1-3 of the dual-slope method to your DataFrame.
    Adds columns: DS_{λ}_AC, DS_{λ}_PH, Δμa_{λ}_AC, Δμa_{λ}_PH for λ=690,830.
    """
    ch_names = [c for c in df.columns if "AC" in c or "PH" in c]
    chs = np.array([deconstruct_ch_name(c) for c in ch_names])
    wls = sorted(list(set([c[2] for c in chs])))
    sources = list(dict.fromkeys([c[0] for c in chs]))
    ch_types = list(dict.fromkeys([c[3] for c in chs]))
    source_pairs = [(sources[i], sources[i + 1]) for i in range(0, len(sources), 2)]
    df_new = pd.DataFrame()
    for ct in ch_types:
        for pair_ix, (s1, s2) in enumerate(source_pairs):
            for wl in wls:
                s1_chs = [c for c in chs if c[0] == s1 and c[2] == wl and c[3] == ct]
                s2_chs = [c for c in chs if c[0] == s2 and c[2] == wl and c[3] == ct]
                s1_names = [construct_ch_name(*c) for c in s1_chs]
                s2_names = [construct_ch_name(*c) for c in s2_chs]
                pair_df = df[s1_names + s2_names]
                delta_mu_a = apply_dualslope_to_pair(pair_df, mu_a, mu_sp)
                # new_ch_name = s1_names[0]
                new_ch_name = f"S{pair_ix + 1}_D{pair_ix + 1} {wl} {ct}"
                print(new_ch_name)
                df_new[new_ch_name] = delta_mu_a
    return df_new


subject = 1
sfreq = 5.2
wl1, wl2 = 690, 830
sub_dir = Path(f"datasets_own/Huang2021/raw_data/sub_{subject}")
csv_files = [sub_dir / f"fnirs_{s}.csv" for s in ["init", "train", "test"]]
df = pd.concat([pd.read_csv(f) for f in csv_files])
# remap values so mne doesn't change them
label_map = {
    0: 1,  # 0-back
    1: 2,  # 1-back
    2: 3,  # 2-back
    3: 4,  # 3-back
    -1: 5,  # rest
}
label2cond = {
    1: "0-back",
    2: "1-back",
    3: "2-back",
    4: "3-back",
    5: "rest",
}
df["label"] = df["label"].map(label_map)
ch_names = [parse_ch_name(c) for c in df.columns[4:] if "AC" in c or "PH" in c]
df.columns = [*df.columns[:4], *ch_names]
df = df.drop(columns=[c for c in df.columns[4:] if "PH" in c])
dfds = apply_dualslope(df)
df_aux = df.iloc[:, :4].reset_index().rename(columns={"index": "original_index"})
df = pd.concat([df_aux, dfds], axis=1)

ch_names = [c for c in df.columns if "AC" in c]
# Hack: mark ch_types as OD to use MBLL
# ch_types = ["fnirs_fd_ac_amplitude"] * len(ch_names)
ch_types = ["fnirs_od"] * len(ch_names)

data = df[ch_names].values.T
ch_names = [c[:-3] for c in ch_names]  # remove type from name
info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(data, info, verbose=False)
# set wl
for ch in raw.info["chs"]:
    wl = ch["ch_name"].split(" ")[1]
    ch["loc"][9] = wl
raw.set_montage(get_montage())
ppf = 0.0001  # picked to visually match the reference data
raw = mne.preprocessing.nirs.beer_lambert_law(raw, ppf=ppf)

filter_kwargs = {
    "l_freq": 0.001,
    "h_freq": 0.2,
    "method": "iir",
    "iir_params": {
        "ftype": "butter",
        "order": 3,
    },
    "verbose": False,
}
raw.filter(**filter_kwargs)

change_ix = np.where((df["label"].diff() != 0))[0]
change_ix = change_ix[1::4]  # We remove rest tasks here, perhaps we could keep them
df["label"].values[change_ix]

events = np.c_[
    change_ix,
    np.zeros(len(change_ix)),
    df["label"].values[change_ix],
].astype(int)
offset = np.round(sfreq * 4).astype(int)  # picked visually to match the reference data
events[:, 0] = events[:, 0] + offset
stm_diffs = np.diff(events[:, 0]) / sfreq
np.c_[events, np.r_[0, stm_diffs]]
u, c = np.unique(events[:, 2], return_counts=True)


def validate_events(events, subject):
    u, c = np.unique(events[:, 2], return_counts=True)
    assert (c == 4).all()
    label = pd.read_csv(f"datasets_own/Huang2021/whole_data/sub_{subject}.csv")["label"]
    label = label[label.shift() != label]
    label = label.map(label_map).values
    assert (label == events[:, 2]).all()
    print(f"Subject {subject} events match.")


validate_events(events, subject)

epochs = mne.Epochs(
    raw, events, tmin=0, tmax=81, baseline=(0, 0), preload=True, verbose=False
)


dfw = pd.read_csv(
    f"datasets_own/Huang2021/whole_data/sub_{subject}.csv"
)  # "whole" data is epoched
ch_map = {
    "S1_D1 hbo": "AB_I_O",
    "S1_D1 hbr": "AB_I_DO",
    "S2_D2 hbo": "CD_I_O",
    "S2_D2 hbr": "CD_I_DO",
}
ch_map_inv = {v: k for k, v in ch_map.items()}
change_ix = np.where(dfw["label"].diff() != 0)[0]
events = np.c_[
    change_ix,
    np.zeros(len(change_ix)),
    dfw["label"].values[change_ix],
].astype(int)

dfw = dfw[ch_map.values()].rename(columns=ch_map_inv)
ch_names = list(dfw.columns)
ch_types = [c.split(" ")[-1] for c in ch_names]
info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(dfw.values.T, info, verbose=False)
epochs_check = mne.Epochs(
    raw, events, tmin=0, tmax=81, preload=True, baseline=(0, 0), verbose=False
)

# %%
epoch_ix = 0
ch_ixs = [0, 1, 2, 3]
scale = 3.1

d1 = epochs.get_data()[epoch_ix][ch_ixs] * scale
d2 = epochs_check.get_data()[epoch_ix][ch_ixs]
ch_names = np.array(epochs.ch_names)[ch_ixs]
ch_names2 = np.array(epochs_check.ch_names)[ch_ixs]
assert (ch_names == ch_names2).all()

# d1 = d1 / (d1.std(axis=1)[:, None] + 1e-6)
# d2 = d2 / (d2.std(axis=1)[:, None] + 1e-6)

fig, axes = plt.subplots(len(ch_ixs), 1, figsize=(10, 3 * len(ch_ixs)), sharex=True)
if len(ch_ixs) == 1:
    axes = [axes]

for i, ch_ix in enumerate(ch_ixs):
    ch_name = ch_names[i]
    # ch_name = ch_map[ch_name] # for original ch names
    ax = axes[i]

    ax.plot(d1[i], color=f"C{i}", linestyle="-", label=f"{ch_name} (from raw)")
    ax.plot(d2[i], color=f"C{i}", linestyle="--", label=f"{ch_name} (reference)")

    ax.set_title(f"Channel: {ch_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Set x-axis labels for the bottom subplot only
every_n = 5
xticks = np.arange(0, d1.shape[1], sfreq * every_n)
xlabels = np.arange(0, d1.shape[1] / sfreq, every_n).astype(int)
axes[-1].set_xlim(0, d1.shape[1])
axes[-1].set_xticks(xticks)
axes[-1].set_xticklabels(xlabels)
axes[-1].set_xlabel("Time (s)")

plt.suptitle(f"Subject {subject}, epoch {epoch_ix}")
plt.tight_layout()
# plt.savefig(f"fnirs2mw.png", dpi=300, bbox_inches="tight")
plt.show()
# %%

# %%
