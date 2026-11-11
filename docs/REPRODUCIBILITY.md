# Reproducibility

The repository provides scientific source code, dataset preparation notebooks, analysis scripts, configuration settings, and software tests for the HemoBoundary study.

## Code and data checks

- `src/morphology.py` is identified by SHA-256 `0e3bfaa393fdcc9a7f648db3079e239824dcd7c968d63a5bac3513a45cc4aaeb`.
- The fourth-dataset manifest lists the 14 selected resting-state recordings, expected channel counts, and SHA-256 checksums.
- Source-model fitting contexts are fixed to preserve the participant-grouped refinement splits.
- The expanded-comparison source snapshot has a separate integrity manifest.
- Python syntax, notebook structure, self-tests, and synthetic test cases can be checked without participant recordings.

These checks establish the integrity of the supplied code and its tested functions. They do not establish complete numerical reproduction from newly downloaded recordings.

## Required inputs and generated outputs

Raw recordings, preprocessed signals, generated test banks, fitted checkpoints, and completed result files are not included in the repository. Analyses save their outputs in user-selected locations outside the repository.

The Shin source evaluation prepares participant-level training banks and selected configurations for downstream analyses. The recorded-background workflow selects unaugmented SNIRF channels and verifies the recordings against the manifest. When the optional reference predictions are supplied, the workflow also checks exact Seed 1 replay consistency.

The recorded-background Cohen-Adad, Lina, and HRfunc adaptations additionally depend on source-only calibration records and fixed reference files from their preceding analyses. The repository does not currently contain a fully standalone reproduction of those reported comparisons. The six-method recorded-background comparison and the ten-seed HemoBoundary robustness test use different source-model designs and must be interpreted separately.

## Statistical inference

The nested classical comparisons use 100,000 two-sided participant-level sign-flip draws and 20,000 participant-level bootstrap draws. Holm correction is applied separately to three families of 32 timing comparisons and one family of eight detection comparisons. The ten-source resting-background analysis averages source models within each of 14 external participants before constructing bootstrap confidence intervals.

The source scripts specify 20,000 sign-flip repetitions for the initial ablation, component perturbation, seven-method benchmark, and external LBR comparisons. The expanded 96-comparison analysis specifies 100,000. Confirm the corresponding execution settings against the saved analyses before using the manuscript's numerical p-values as evidence of a completed rerun.

## Verification scope

A complete end-to-end reproduction of every manuscript result from independently obtained public datasets has not been verified. Statistical summaries require the corresponding output files, and the published-method adaptations described above require additional calibration inputs. These limits do not affect the distinction between the primary known-boundary tests and the separate recorded-background transfer experiment.
