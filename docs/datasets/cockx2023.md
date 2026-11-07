# Cockx2023

- Subjects: 24
- Epochs: 1844
- Classes: 2 (Finger, Foot)
- Task: Finger vs Foot (multi-task tapping)
- Task type: Motor (multi-task)
- LOSO Acc: 85.4%
- Within Acc: 94.7%
- Description: Cockx2023 fNIRS dataset (BIDS format with special channel mapping)
- Paper: Automatic and non-automatic finger and leg movements measured with functional near-infrared spectroscopy (fNIRS)
- Paper link: https://www.sciencedirect.com/science/article/pii/S1053811923000290?via%3Dihub
- Data link: https://data.ru.nl/collections/di/dcn/DSC_62003212_03_066
- Device: Artinis Brite24
- n_channels: 24
- sfreq: 50 Hz
- Montage: yes, 3D
- Data format: BIDS
- Original data type: fnirs_od 
- LICENSE: CC-BY-SA-4.0

## Task details

Participants tapped four right-hand fingers on a keyboard or tapped the right foot on four squares on the floor in a specific order given by a 12-digit sequence. Two different sequences (A, B) were learned: one practiced for 7 days (automatic), the other only on the experiment day (non-automatic). Each task was performed under single- and dual-task conditions (dual-task: count 'G's in a random letter stream during tapping).

- 8 tasks per subject
- 11 trials per task (in a row)
- Epoch: 8s metronome, 12-15s baseline, 8s task, 5s baseline, (survey for dual)
- Dual-task: count 'G's in random letter stream during tapping

## Citation

@article{cockx2023fnirs,
  title={fNIRS is sensitive to leg activity in the primary motor cortex after systemic artifact correction},
  author={Cockx, Helena and Oostenveld, Robert and Tabor, Merel and Savenco, Ecaterina and van Setten, Arne and Cameron, Ian and van Wezel, Richard},
  journal={NeuroImage},
  volume={269},
  pages={119880},
  year={2023},
  publisher={Elsevier}
}
