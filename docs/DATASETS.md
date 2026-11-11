# Datasets and preprocessing

The recordings are not distributed with this code. Obtain each dataset from its original provider and follow the provider's conditions.

| Dataset | Public reference | Local prepared data |
|---|---|---|
| Shin, word generation | https://doi.org/10.1038/sdata.2018.3 | `VP001_WG_ALIGNED_PREPROCESSED.npz` through `VP026_WG_ALIGNED_PREPROCESSED.npz` |
| Bálint, speech comprehension | https://doi.org/10.5061/dryad.crjdfn3g9 | `balint_epochs_10hz.h5` |
| Khan, finger movement | https://doi.org/10.3389/fnhum.2026.1747655 | `khan2026_finger_epochs_10hz.h5` |
| von Lühmann, resting-state Dataset I | https://doi.org/10.3389/fnins.2020.579353 | Original unaugmented channels identified within 14 checksum-matched SNIRF recordings |

## Google Colab layout

Existing scientific source scripts use the original project cache layout. The public notebooks use these folders by default:

```text
MyDrive/HonorProject/
├── shin2018_word_generation_state_preparation_v1/
│   ├── 02_preprocessed/
│   └── [source bank and selected-configuration caches]
└── HemoBoundary/
    ├── SHIN/
    ├── Balint/
    ├── Khan/
    └── Luhmann2020_Resting/
```

The first three preprocessing notebooks prepare the main task datasets. Shin uses the provided concentration signals and a causal 0.2 Hz low-pass branch for the tested noise profiles. Bálint uses short-separation regression, band-pass filtering, three-wavelength conversion, and TDDR. Khan uses optical density, TDDR, modified Beer-Lambert conversion, and filtering. The original authors' preprocessing methods are not assumed to match these analysis pipelines exactly.

## Fourth dataset

Run `preprocessing/04_von_Luhmann_Resting.ipynb` to obtain and check Dataset I. A NITRC provider download is available in the notebook. If provider access or the archive URL has changed, obtain the files manually from the publisher and place them under `Luhmann2020_Resting/`.

The fourth-dataset pipeline identifies original measurement pairs with `dataTypeLabel=0`, which excludes previously injected channels. It converts raw optical intensity to optical density and then to HbO/HbR concentrations with a partial pathlength factor of 6. The concentration signals are filtered from 0.01 to 0.2 Hz and resampled to 2 Hz. Four non-overlapping windows, starting 30 s apart, are selected from each participant. The fixed selection of 14 source files and their SHA-256 checksums is recorded in `docs/RECORDED_BACKGROUND_FILES.json`.

For each of 56 windows, one control and three generated-response conditions are created. The HbO injection levels are 0.5, 1.0, and 2.0 times the recorded baseline standard deviation, with channel-specific variation and an opposing HbR response. This gives 168 injected responses and 56 controls. The responses are generated, but their backgrounds are real recordings. The deterministic response-generation code is in `src/reproduction_utils.py`.

The recorded-background study is a separate transfer stress test. It is not interchangeable with the primary generated-noise evaluation or a test with independently annotated natural-response boundaries.

Provider access tokens must be supplied privately in the runtime. Do not commit raw data, recordings, generated banks, or prediction outputs to GitHub.

The additional Colab notebooks check the two documented Shin cache roots directly. They use `HonorProject/shin2018_word_generation_state_preparation_v1` when it contains the required source banks, and otherwise use `HonorProject/HemoBoundary/SHIN`. They do not search unrelated Drive folders or silently substitute different test banks.
