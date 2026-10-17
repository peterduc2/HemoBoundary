# HemoBoundary

HemoBoundary is a framework for detecting trial-level fNIRS responses and estimating their onset, primary peak, and recovery boundaries. It combines an ordered temporal-state representation, multiscale waveform features, response detection, and local boundary refinement.

## Repository structure

| Folder | Contents |
|---|---|
| `preprocessing/` | Shin, Bálint, and Khan preprocessing notebooks |
| `analysis/` | Seven analysis notebooks for the main experiments |
| `src/` | Reusable models and analysis code, including the expanded comparisons |
| `scripts/` | Experiment runners and verification utilities |
| `docs/` | Dataset preparation, evaluation settings, and execution instructions |

Python implementations are maintained in `src/`. Use the executable notebooks under `preprocessing/` and `analysis/` for dataset preparation and experiments.

## Getting started

1. [Obtain and prepare the datasets](docs/DATASETS.md).
2. Run the three notebooks in `preprocessing/`.
3. Follow the [analysis run order](docs/RUN_ORDER.md).
4. Run `python scripts/audit_structure.py` to verify repository structure and source integrity.
5. For additional methods and sensitivity analyses, follow the [expanded-comparison guide](docs/EXPANDED_COMPARISON.md).

The main evaluation uses leave-one-subject-out testing across 26 Shin participants and 10 seeds, with model selection restricted to source participants. Statistical inference is conducted at the participant level after averaging across seeds. Timing accuracy is evaluated using generated responses with known boundaries and participant-derived noise. Analysis of recovery in recorded task responses is separate.

## Data, code, and verification

The code for the core evaluation, seven timing references, eight additional comparators, and prediction-based sensitivity analyses is included. Participant recordings, generated test banks, saved predictions, and checkpoints are not distributed. The original expanded-comparison source archive and independently checked source hashes are retained for provenance.

Source-integrity checks and available software self-tests have passed. **Complete numerical reproduction from independently downloaded datasets has not been verified.** Some analyses require the original seed-specific checkpoint structure and saved-prediction exports. See [reproducibility status](docs/REPRODUCIBILITY.md) for details.

Code is released under the MIT license. Please consult the original dataset providers for access and reuse conditions, and see `CITATION.cff` for citation information.
