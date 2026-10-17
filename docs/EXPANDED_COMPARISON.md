# Expanded comparison and sensitivity analysis

The methods used for the expanded comparisons in the manuscript are included as source code. They are task-specific adaptations, not exact reproductions of the original published continuous-recording methods.

## Source files

- `src/expanded/frozen/` contains the frozen scientific implementation for ExtraTrees, Ridge, HRfunc, and additional original comparison methods.
- `src/expanded/cohen_adad.py` contains the two Cohen-Adad-inspired GLM implementations.
- `src/expanded/lina_wavelet.py` contains the wavelet implementations with and without PCA.
- `src/expanded/paired_statistics.py` computes the participant-paired contrasts for the expanded comparison.
- `src/expanded/sensitivity.py` rescales missing-estimate penalties and evaluates conditional and common-valid MAE and coverage from saved predictions.
- `src/expanded/sensitivity_review.py` checks the completed sensitivity results.

The exact frozen source archive is preserved under `docs/provenance/frozen_comparison_source.zip`, with SHA-256 `a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d`. The archive and all 21 original members are checked against their recorded hashes. The repository-facing README in the extracted source folder links to the current instructions, while the original README remains unmodified inside the archived ZIP. The Cohen-Adad and Lina scripts have also been checked against their original recorded code hashes. The scientific files were not reformatted or modified.

## Quick source checks

Run from the repository root:

```bash
python scripts/run_expanded_comparison.py check
python scripts/run_expanded_comparison.py cohen --mode selftest
python scripts/run_expanded_comparison.py lina --mode selftest
python scripts/run_expanded_comparison.py paired --mode selftest
python scripts/run_expanded_comparison.py sensitivity --mode selftest
```

## Reproduction with the original datasets

The historical experiment uses the Shin, Bálint, and Khan prepared recordings, participant-specific generated banks, original prediction checkpoints, and a 770-case verified export. These data and derived results are **not committed** to GitHub. The original output locations and protocol identifiers remain inside the source to allow saved-prediction parity checks. Do not rename those identifiers without migrating the corresponding outputs.

The original benchmark source can run from the repository with:

```bash
python scripts/run_expanded_comparison.py frozen --mode full
```

This uses the original isolated-environment launcher and requires the dataset preparation and saved-result paths described in this guide and [dataset setup](DATASETS.md). It can be resumed by the original benchmark's checkpointing logic. For a small initial run, use `--mode pilot`.

The separate source-calibrated adaptations can be invoked in Google Colab after the required benchmark results have been prepared:

```bash
python scripts/run_expanded_comparison.py cohen --mode full
python scripts/run_expanded_comparison.py lina --mode full
```

The runner stages only a verified copy of the frozen source archive under the archived scripts' expected Colab filenames. It does not overwrite the original prediction outputs. These adaptations require the source-seed result folders for seeds 1–5 and 6–10, together with their verified protocol files. They create their own result directories.

The paired statistics and sensitivity analyses then use **saved original results**, not a new model fit:

```bash
python scripts/run_expanded_comparison.py paired --mode full
python scripts/run_expanded_comparison.py sensitivity --mode full
python scripts/run_expanded_comparison.py review --mode full
```

These scripts require the saved checkpoint trees for seeds 1–5 and 6–10, the final verified 770-case export, and completed Cohen-Adad and Lina outputs. Running the frozen package in a new output folder does not automatically populate all those historical directories. An identical end-to-end numerical rerun from fresh datasets has **not** been verified. Until this is done, the recovered files prove code identity but not independent reproduction of Table 4 or Tables S16–S18.
