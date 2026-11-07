# BNCI

- Subjects: 8
- Epochs: 348
- Classes: 2 (MA, BL)
- Task: Mental Arithmetic (MA) vs Baseline (BL)
- Task type: Mental Arithmetic (MA)
- LOSO Acc: 80.5%
- Within Acc: 98.5%
- Description: BNCI Horizon 2020 Dataset Mental arithmetic (003-2014)
- Paper: BNCI horizon 2020, 3. Mental arithmetic (003-2014) (Single-trial classification of antagonistic oxyhemoglobin responses during mental arithmetic)
- Paper link: https://link-springer-com.libproxy.aalto.fi/article/10.1007/s11517-011-0792-5
- Data link: https://bnci-horizon-2020.eu/database/data-sets
- Device: Hitachi ETG-4000
- n_channels: 52
- sfreq: 10 Hz
- Montage: Custom, made from the figure + description in the paper. Quality estimation 8/10.
- Data format: mat
- Original data type: hbo_hbr 
- is_filtered: False
- LICENSE: CC BY-ND 4.0

## Experimental paradigm

Participants performed cue-guided mental calculations: sequentially subtracting a one-digit number from a two-digit number as quickly as possible for 12 s (e.g. 97-4=93, 93-4=89, ...), followed by 28 s of rest. Participants performed 18 or 24 trials of MA, respectively.

**Note:** In this version, an 8 s separation was added between the end of the MA epoch and the start of the rest epoch (original: 12s MA → 28s rest; modified: 12s MA + 8s space → 20s rest).

## Citation

@article{bauernfeind2011single,
  title={Single-trial classification of antagonistic oxyhemoglobin responses during mental arithmetic},
  author={Bauernfeind, G{\"u}nther and Scherer, Reinhold and Pfurtscheller, Gert and Neuper, Christa},
  journal={Medical \& biological engineering \& computing},
  volume={49},
  number={9},
  pages={979--984},
  year={2011},
  publisher={Springer}
}
