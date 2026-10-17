# Run order

Run the three preprocessing notebooks first, then the analysis notebooks in the order below. The experiment notebooks handle the associated Python source modules and write results outside the repository.

## Preprocessing

1. `preprocessing/01_SHIN_Preprocessing.ipynb`
2. `preprocessing/02_Balint_Preprocessing.ipynb`
3. `preprocessing/03_Khan_Preprocessing.ipynb`

## Analysis

1. `analysis/01_HemoBoundary_Main.ipynb` — core evaluation with ten seeds and participant checkpoints
2. `analysis/02_Component_Ablation.ipynb` — cumulative model summary
3. `analysis/03_Functional_Necessity.ipynb` — component neutralization
4. `analysis/04_Seven_Method_Benchmark.ipynb` — timing references
5. `analysis/05_Unseen_Morphology_Robustness.ipynb` — morphology robustness
6. `analysis/06_Balint_Khan_External_Validation.ipynb` — external datasets
7. `analysis/07_Late_Window_Recovery.ipynb` — recorded-response recovery

## Commands

```bash
python scripts/audit_structure.py
python scripts/run_core.py --self-test-only
python scripts/run_core.py --shin-root /path/to/SHIN --prepare-initial-bank
python scripts/run_experiment.py benchmark --shin-root /path/to/SHIN
python scripts/run_experiment.py external --shin-root /path/to/SHIN --balint-h5 /path/to/balint_epochs_10hz.h5 --khan-h5 /path/to/khan2026_finger_epochs_10hz.h5
```

Completed seed/participant checkpoints are reused when supported by the experiment. The saved-results checker reads existing results and does not retrain models.

## Expanded comparisons and sensitivity

These analyses are provided as additional source-code workflows and do not alter the seven original main-analysis notebooks. See [expanded comparison](EXPANDED_COMPARISON.md) for the original source requirements, supported commands, output dependencies, and reproducibility limitations.

```bash
python scripts/run_expanded_comparison.py check
python scripts/run_expanded_comparison.py cohen --mode selftest
python scripts/run_expanded_comparison.py lina --mode selftest
python scripts/run_expanded_comparison.py paired --mode selftest
python scripts/run_expanded_comparison.py sensitivity --mode selftest
```
