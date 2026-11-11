# Execution order

Prepare the datasets first, run the source evaluation, then run the remaining analyses. Recordings and generated results remain outside the repository.

## Data preparation

1. `preprocessing/01_SHIN_Preprocessing.ipynb`
2. `preprocessing/02_Balint_Preprocessing.ipynb`
3. `preprocessing/03_Khan_Preprocessing.ipynb`
4. `preprocessing/04_von_Luhmann_Resting.ipynb`

## Primary analyses

1. `analysis/01_HemoBoundary_Main.ipynb`
2. `analysis/02_Component_Ablation.ipynb`
3. `analysis/03_Functional_Necessity.ipynb`
4. `analysis/04_Seven_Method_Benchmark.ipynb`
5. `analysis/05_Unseen_Morphology_Robustness.ipynb`
6. `analysis/06_Balint_Khan_External_Validation.ipynb`
7. `analysis/07_Late_Window_Recovery.ipynb`

The Shin source evaluation prepares the seed-specific training banks and selected configuration files required by subsequent experiments.

## Additional analyses

8. `analysis/08_Recorded_Background.ipynb` - recorded-background evaluation and ten-source robustness
9. `analysis/09_Nested_Classical.ipynb` - participant-grouped nested Ridge and ExtraTrees comparisons
10. `analysis/10_Transfer_Diagnostics.ipynb` - response-score, threshold, and surrogate-background analyses

In Google Colab, these notebooks use the local repository at `/content/HemoBoundary` when present, or clone the public GitHub repository. Ensure that the selected code revision contains the analyses to be run. The notebooks use `/content/gdrive/MyDrive/HonorProject` as their default project root.

## Software checks

From the repository root:

```bash
python scripts/audit_structure.py
python scripts/run_core.py --self-test-only
python scripts/run_recorded_background.py selftest
python scripts/run_nested_classical.py --self-test
python scripts/run_transfer_diagnostics.py selftest
python scripts/run_expanded_comparison.py check
```

## Running additional experiments

Provide paths to the prepared Shin cache, the resting-state SNIRF files, and an external output directory:

```bash
python scripts/run_recorded_background.py inspect --rest-root /path/to/rest --output /path/to/results
python scripts/run_recorded_background.py run --shin-root /path/to/shin_cache --rest-root /path/to/rest --output /path/to/results
python scripts/run_nested_classical.py --shin-root /path/to/shin_cache --output /path/to/results
python scripts/run_transfer_diagnostics.py all --shin-root /path/to/shin_cache --rest-root /path/to/rest --completed /path/to/results --output /path/to/diagnostics
```

The recorded-background runner can obtain Dataset I from the provider when `--download` is supplied. To verify the saved Seed 1 predictions, provide `--original-archive /path/to/reference.zip`. The nested analysis also accepts an optional reference archive for consistency checks.

## Published-method comparisons on recorded backgrounds

The Cohen-Adad, Lina, and HRfunc modules implement task-specific adaptations. They require source-calibration records and reference files prepared by preceding workflows. Once those inputs are available, run:

```bash
python scripts/run_recorded_published.py all --project-root /path/to/project --output /path/to/results/adapters
```

The HRfunc adapter requires `hrfunc==1.3.1`. Source-calibration identifiers are retained where necessary to reproduce the selected settings. The three primary known-boundary evaluations, recorded-background comparisons, ten-source robustness test, and nested classical analyses have different statistical designs and should be reported separately.
