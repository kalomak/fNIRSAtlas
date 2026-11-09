# Glossary

Always use same word when referring to the same concept, in docs and in code.

## Data

- **Subject**: one participant, an integer id in `dataset.subject_list`.
- **Session**: one visit of a subject. Keys of `dataset.get_data()[subject]`.
- **Run**: one continuous recording within a session, one MNE `Raw`. Keys of
  `dataset.get_data()[subject][session]`. Written `subject/session/run`, e.g.
  `27/0/4silence`.
- **Source-detector group**: the channels sharing a source and detector, one
  per wavelength or chromophore, e.g. `S1_D1 760` and `S1_D1 850`. CW and OD
  groups can have more than two wavelengths. 
- **Epoch**: the segment around one event, the dataset `interval` in seconds
  relative to onset.

## Benchmark

- **Paradigm**: turns runs into labelled epochs, e.g. `FNIRSClassification`.
- **Pipeline**: sklearn-compatible model applied to the epochs.
- **Evaluation**: MOABB cross-validation scheme, e.g. `CrossSubject` (LOSO,
  leave one subject out) or `WithinSession`.
