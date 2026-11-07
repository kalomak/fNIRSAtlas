# Guevara2024

- Subjects: 20 patients with Parkinson’s disease, 20 controls
- Tasks:
  - Tapping: Left/Right finger tapping (Motor)
  - Walking: Over-ground walk (Motor/Gait)
  - Resting-State: Eyes-closed rest (Resting-state/FC)
- Paper: Parkinson’s disease patients show delayed hemodynamic changes in primary motor cortex in fine motor tasks and decreased resting-state interhemispheric functional connectivity: a functional near-infrared spectroscopy study
- Paper link: https://www.spiedigitallibrary.org/journals/neurophotonics/volume-11/issue-02/025004/Parkinsons-disease-patients-show-delayed-hemodynamic-changes-in-primary-motor/10.1117/1.NPh.11.2.025004.full#_=_
- Data link: https://zenodo.org/records/7966830
- Device: Artinis Brite MKII
- n_channels: 20
- sfreq: 25 Hz (varies: 25, 50, 75)
- Montage: 10 sources, 8 detectors, 3D digitized, 3 cm (long), 1.5 cm (short)
- Data format: snirf
- Original data type: mne
- LICENSE: CC-BY-4.0

## Task Details

### Tapping Task
- 10 s thumb-to-finger tapping (left or right hand, 10 blocks each, randomized)
- Rest: 20–24 s between blocks
- Video cues for movement onset/offset
- 20 blocks per subject (10 left, 10 right)

Participants performed a fine‑motor finger‑tapping paradigm lasting approximately 11 min, divided into 20 randomized blocks (10 with the left hand, 10 with the right). Each block consisted of a 10 s burst of thumb‑to‑finger tapping, cued by video instructions for movement onset and offset, followed by a variable rest interval of 20–24 s to prevent phase‑locking to physiological oscillations and reduce anticipatory effects .

### Walking Task
- 2 min over-ground walk, 20 s standing baseline
- Walk back and forth on 6 m path, 180° turns
- Hemodynamic response analyzed from 2–25 s after walk onset

Participants performed a 2‑minute 
over‑ground walk preceded by a 20 s 
standing baseline. They walked back 
and forth along a 6 m straight path, 
making 180° turns at each end. The 
hemodynamic response was analyzed as 
a single trial from –2 s to 25 s 
relative to walk onset, but the full 
2 min were recorded to detect any 
gait freezing. A researcher walked 
beside each participant for safety 
and to assist during turns .

### Resting-State (RSFC) Task
- 6 min eyes-closed, seated, motionless
- Monitored to prevent sleep/movement

Subjects sat quietly, motionless, in 
a dimly lit room for 6 min with their 
eyes closed and were instructed not 
to focus on anything in particular. A 
team member observed them to prevent 
movement or dozing, and afterward 
confirmed that none had fallen 
asleep. These clean rest data were 
then used for functional connectivity 
analysis .

## Citation

@article{guevara2024parkinson,
  title={Parkinson’s disease patients show delayed hemodynamic changes in primary motor cortex in fine motor tasks and decreased resting-state interhemispheric functional connectivity: a functional near-infrared spectroscopy study},
  author={Guevara, Edgar and Rivas-Ruvalcaba, Francisco Javier and Kolosovas-Machuca, Eleazar Samuel and Ram{\'\i}rez-El{\'\i}as, Miguel and de Le{\'o}n Zapata, Ram{\'o}n D{\'\i}az and Ramirez-GarciaLuna, Jose Luis and Rodr{\'\i}guez-Leyva, Ildefonso},
  journal={Neurophotonics},
  volume={11},
  number={2},
  pages={025004--025004},
  year={2024},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
