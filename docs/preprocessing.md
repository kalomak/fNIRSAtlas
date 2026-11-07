# Preprocessing Pipeline

## Standard Steps
1. Short channel regression (optional)
2. Remove short channels (optional)
3. SCI-based bad channel detection
4. Channel interpolation
5. TDDR motion correction
6. Optical density → HbO/HbR conversion
7. Band-pass filtering (0.01–0.1 Hz)

## Dataset-Specific Deviations
- **Cockx2023**: Short channel threshold 0.02
- **Shin2019**: Channel filling (1 subject is missing one channel)