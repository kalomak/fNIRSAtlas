# %%
"""
To confirm our suspicion that the subject Control18 in Guevara2024 has wrong sfreq, we:
1. Check stimulus intervals
2. Check PSD
3. Check recording length

Correct intervals according to the paper:
"10s burst of tapping, rest periods lasting between 20s and 24s were used"
So about 30s.
"""

from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np

from fnirsatlas.utils import override_raw_sfreq

# %%

adjust_sfreq = False
sub1 = "18"
sub2 = "04"  # "Normal" subject with the same marked sfreq (25Hz)
experiment = "finger_tapping"
experiments = ["resting_state", "finger_tapping", "walking"]
experiment_idx = experiments.index(experiment)
root_dir = Path("datasets/Guevara2024/fNIRS_Parkinson")
snirf_path = (
    root_dir
    / f"{experiment}/Control/Control{sub1}/{experiment_idx + 1}_{experiment}.snirf"
)
snirf1 = mne.io.read_raw_snirf(snirf_path, verbose=False, preload=True)

if adjust_sfreq:
    snirf1 = override_raw_sfreq(snirf1, factor=0.5)
    print(f"Adjusted sfreq for {snirf_path} to {snirf1.info['sfreq']} Hz")

snirf_path = (
    root_dir
    / f"{experiment}/Control/Control{sub2}/{experiment_idx + 1}_{experiment}.snirf"
)
snirf2 = mne.io.read_raw_snirf(snirf_path, verbose=False, preload=True)

# %%
onsets1 = np.unique(snirf1.annotations.__getattribute__("onset"))
onsets2 = np.unique(snirf2.annotations.__getattribute__("onset"))

intervals1 = np.diff(onsets1)
intervals2 = np.diff(onsets2)

print(f"C18: {intervals1.mean():.2f}s, Normal: {intervals2.mean():.2f}s")
# Evidence 1: Stimulus intervals are about half of the usual length
# %%
d = snirf1.get_data()
d2 = snirf2.get_data()


sfreq1, sfreq2 = snirf1.info["sfreq"], snirf2.info["sfreq"]
seconds1, seconds2 = d.shape[1] / sfreq1, d2.shape[1] / sfreq2
print(f"C18: {seconds1:.2f}s, Normal: {seconds2:.2f}s")
# Evidence 2: C18 recording length is about half of other recordings
# %%
fig = snirf1.compute_psd(fmax=6).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)
fig.suptitle(f"C18 ({sub1})")
plt.show()

fig = snirf2.compute_psd(fmax=6).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)
fig.suptitle(f"Normal ({sub2})")
plt.show()
# Evidence 2: C18 has the same pattern, but spikes at about 2x freqs
# %%
