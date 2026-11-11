# HemoBoundary

HemoBoundary estimates trial-level fNIRS response onset, primary peak, and recovery using ordered temporal-state features, multiscale waveform features, response detection, and local boundary refinement.

## Data and evaluation

The primary evaluations use generated responses with known temporal boundaries and noise profiles estimated from the Shin, Bálint, and Khan recordings. An additional transfer stress test injects generated responses into recorded resting-state fNIRS backgrounds from 14 participants in the von Lühmann dataset. The external test does not provide independently annotated natural-response boundaries.

## Repository structure

| Folder | Contents |
|---|---|
| `preprocessing/` | Preparation notebooks for four public datasets |
| `analysis/` | Main experiments, comparisons, and transfer analyses |
| `src/` | Model implementations, evaluation methods, and statistical analyses |
| `scripts/` | Execution commands and validation checks |
| `docs/` | Dataset information, configuration settings, and reproducibility instructions |

## Getting started

1. Obtain the recordings from their official providers and follow [dataset preparation](docs/DATASETS.md). Recordings are not included in this repository.
2. Run the preparation notebooks in `preprocessing/`.
3. Follow the sequence in [run order](docs/RUN_ORDER.md), beginning with the Shin source analysis.
4. Run the [software checks](docs/RUN_ORDER.md#command-line-checks) before the full evaluations.
5. Review the [reproducibility notes](docs/REPRODUCIBILITY.md) for required inputs and verification limits.

The analyses include source-nested Ridge and ExtraTrees comparisons, recorded-background evaluation across ten Shin source seeds, and exploratory transfer diagnostics. Comparisons with published fNIRS approaches use task-specific adaptations and should not be interpreted as direct reproductions of each method's original application.

## Reproducibility

The code includes deterministic procedures, configuration records, and software tests. Participant recordings, prepared datasets, generated response banks, trained checkpoints, and completed experimental outputs are not distributed here. Certain published-method adaptations require source-calibration artifacts that are not available in this repository.

Software and source-integrity checks are available, but complete end-to-end numerical reproduction from independently downloaded public datasets has not been confirmed. See [reproducibility details](docs/REPRODUCIBILITY.md).

The code is distributed under the MIT license. Dataset access and reuse are governed by the original providers. Citation information is provided in `CITATION.cff`.
