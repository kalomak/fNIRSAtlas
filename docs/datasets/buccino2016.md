# Buccino2016

- Subjects: 15
- Epochs: 1500
- Classes: 4 (left_arm, left_hand, right_arm, right_hand)
- Task: Right-Arm—Left-Arm raising, and Right-Hand—Left-Hand gripping
- Task type: Motor execution
- LOSO Acc: 46.2%
- Within Acc: 67.8%
- Description: Buccino2016 EEG-fNIRS hybrid SMR BCI dataset
- Paper: Hybrid EEG-fNIRS Asynchronous BrainComputer Interface for Multiple Motor Tasks
- Paper link: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0146610&type=printable
- Data link: https://figshare.com/articles/dataset/EEG_fNIRShybrid_SMR_BCI_data_S1_S8/1619640?file=2601537, https://figshare.com/articles/dataset/EEG_fNIRS_hybrid_SMR_BCI_data_S9_S15/1619641?file=2601481
- Device: NIRx NIRScout 8-16
- n_channels: 36
- sfreq: 10.42 Hz
- Montage: picture
- Data format: mat
- Original data type: fnirs_cw_amplitude 
- LICENSE: CC BY 4.0

## Task details

- 15 healthy right-handed male subjects
- The experiment consisted of 5 blocks of motor execution:
in each block, subjects were to perform 20 trials divided in the 4 movements (Right-Arm—
Left-Arm raising, and Right-Hand—Left-Hand gripping). The trials for each block were randomized, but the number of trials were evenly distributed, so that for each block 5 trials for
each class were acquired. In total, 25 trials for each class were performed by each subject. The
subjects were guided through the experiment following the visual instructions (textual) presented on a laptop screen placed around 1 m away from their eyes. Every trial started with 6
seconds of rest, and the subject was instructed for the following 6 seconds to move according to
the screen direction (Right Arm, Left Arm, Right Hand, Left Hand) at a self-paced rhythm.
Every trial lasted 12 seconds and the actual data acquisition was about 20 minutes excluding
the instrumentation setup. Simultaneous EEG and fNIRS measurements were acquired during
the experiment. The fNIRS system (NIRScout 8–16, NIRx Medizintechnik GmbH, Germany)
was equipped with 12 sources and 12 detectors combined in 34 channels of acquisition. The
channels were distributed evenly on the motor cortex, the sampling frequency was fnirs = 10.42
Hz, and the wavelength used were 760 nm and 850 nm. The EEG system (microEEG, BioSignal
Group, US) was used with 21 measurement channels (F3, Fz, F4, Fc5, Fc1, Fc2, Fc6, C5, C3,
C1, Cz, C2, C4, C6, Cp5, Cp1, Cp2, Cp6, P3, Pz, and P4) referenced to Fcz. The ground electrode was placed frontally on Fpz. The electrodes used were standard Ag/AgCl ones and the
EEG signals were sampled at feeg = 250 Hz. EEG electrodes and fNIRS probes were mounted on
an extended EEG cap (actiCAP 128, Brain Products GmbH, Germany) and fNIRS sources and
detectors were placed maximum at 3.4 cm from each other, in order to ensure good quality signals. The presence of hair, that represents the main limitation for fNIRS recordings, was treated
with a cautious placement of the optodes after moving the hair aside with optically conductive
gel. NIRStar software (NIRx Medizintechnik GmbH, Germany) was used both to acquire the
data and to check the quality of the signals before starting the experiments. Presentation software (Neurobehavioral Systems, US) was used to guide the subjects during the experiment, to
synchronize the signals, and to keep the log of the data.

- 25 trials per class per subject (4 classes: Right-Arm, Left-Arm raising, Right-Hand, Left-Hand gripping)
- Each trial: 6 s rest, 6 s movement (self-paced, per screen direction), total 12 s
- 5 blocks per subject, 20 trials per block (5 per class, randomized)
- Visual instructions on screen, 1 m from subject
- Simultaneous EEG and fNIRS (34 fNIRS channels, 21 EEG channels)
- fNIRS: NIRScout 8–16, 12 sources, 12 detectors, 34 channels, 10.42 Hz, 760/850 nm
- EEG: microEEG, 21 channels, 250 Hz, standard Ag/AgCl, referenced to Fcz
- Optodes placed max 3.4 cm apart, hair moved aside for signal quality
- NIRStar software for acquisition, Presentation software for instructions/logs

## Expected performance

- Right-left fNIRS classification (LDA): 63.1±5.8
- Arm-hand classification: 80.4±9.1

## Citation

@article{buccino2016hybrid,
  title={Hybrid EEG-fNIRS asynchronous brain-computer interface for multiple motor tasks},
  author={Buccino, Alessio Paolo and Keles, Hasan Onur and Omurtag, Ahmet},
  journal={PloS one},
  volume={11},
  number={1},
  pages={e0146610},
  year={2016},
  publisher={Public Library of Science San Francisco, CA USA}
}
