# Datasets and preprocessing

The dataset files are not included. Obtain data only through the respective public providers and comply with their terms of access.

| Dataset | Reference | Prepared output |
|---|---|---|
| Shin, word generation | https://doi.org/10.1038/sdata.2018.3 | `VP001_WG_ALIGNED_PREPROCESSED.npz` through `VP026_WG_ALIGNED_PREPROCESSED.npz` |
| Bálint, speech comprehension | https://doi.org/10.5061/dryad.crjdfn3g9 | `balint_epochs_10hz.h5` |
| Khan, finger movement | https://doi.org/10.3389/fnhum.2026.1747655 | `khan2026_finger_epochs_10hz.h5` |

## Colab project root

The main notebook convention is `/content/gdrive/MyDrive/HonorProject/HemoBoundary/`. The source scientific runners retain the existing output directories for compatibility with saved results.

```text
MyDrive/HonorProject/HemoBoundary/
├── SHIN/
│   ├── 01_download/raw_matlab/
│   ├── 02_preprocessed/
│   └── [generated experiment results, local only]
├── Balint/
│   ├── raw/
│   └── processed/balint_epochs_10hz.h5
└── Khan/
    ├── raw_download/
    └── processed/khan2026_finger_epochs_10hz.h5
```

Shin input MATLAB files contain recorded HbO/HbR at 10 Hz. The selected causal 0.2-Hz branch is used to estimate noise profiles. Bálint uses short-separation regression, filter, three-wavelength conversion and TDDR. Khan converts CW amplitude to optical density, applies TDDR, MBLL and temporal filtering before resampling to 10 Hz. Consult the preprocessing notebooks for full detail.

Private provider share tokens are not committed. If you are authorized to access a Khan private share, set `HEMOBOUNDARY_KHAN_FIGSHARE_TOKEN` privately in the Colab runtime. Alternatively, download the provider archive manually and place it in the expected raw-data directory.

The source code for the eight additional comparators and the prediction-based sensitivity analyses is included in `src/expanded/`. Reproducing the published results also requires the corresponding input recordings, generated test banks, participant-level checkpoints, and the saved 770-case export. See [expanded comparison](EXPANDED_COMPARISON.md) for the required workflow.
