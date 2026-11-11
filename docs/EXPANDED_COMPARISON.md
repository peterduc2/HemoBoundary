# Expanded comparison and sensitivity analysis

The expanded comparisons use task-specific adaptations of established fNIRS methods. They are not complete reproductions of the methods in their original application settings.

## Source files

- `src/expanded/frozen/` contains the source implementation for ExtraTrees, Ridge, HRfunc, and additional comparison methods.
- `src/expanded/cohen_adad.py` implements two GLM adaptations based on Cohen-Adad et al.
- `src/expanded/lina_wavelet.py` implements wavelet-based analyses with and without PCA.
- `src/expanded/paired_statistics.py` computes participant-level paired comparisons.
- `src/expanded/sensitivity.py` evaluates missing-estimate penalties, conditional and common-valid mean absolute error, and prediction coverage.
- `src/expanded/sensitivity_review.py` checks the resulting sensitivity summaries.

A source snapshot is provided at `docs/provenance/frozen_comparison_source.zip` with SHA-256 `a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d`. The verification command checks its member hashes and the preserved scientific source files. These identifiers are retained to support reproducibility.

## Source checks

From the repository root, run:

```bash
python scripts/run_expanded_comparison.py check
python scripts/run_expanded_comparison.py cohen --mode selftest
python scripts/run_expanded_comparison.py lina --mode selftest
python scripts/run_expanded_comparison.py paired --mode selftest
python scripts/run_expanded_comparison.py sensitivity --mode selftest
```

## Running the benchmark

The benchmark requires prepared Shin, Bálint, and Khan data, generated response banks, selected source configurations, and saved predictions for the later inference stages. These inputs are stored outside the repository.

The source benchmark can be started with:

```bash
python scripts/run_expanded_comparison.py frozen --mode preflight
python scripts/run_expanded_comparison.py frozen --mode full
```

The benchmark supports checkpoint-based continuation. To perform a limited protocol check, use `--mode pilot`.

Once the required source results are available, run the two additional method adaptations:

```bash
python scripts/run_expanded_comparison.py cohen --mode full
python scripts/run_expanded_comparison.py lina --mode full
```

The reference source archive is placed under the filenames expected by these analyses. The source-data and selected-configuration identifiers must remain unchanged for valid comparisons with saved results.

Finally, compute the paired statistical comparisons and sensitivity summaries:

```bash
python scripts/run_expanded_comparison.py paired --mode full
python scripts/run_expanded_comparison.py sensitivity --mode full
python scripts/run_expanded_comparison.py review --mode full
```

These stages require the completed seed groups (1-5 and 6-10), the 770-case benchmark export, and the relevant method outputs. Source integrity checks do not, by themselves, establish numerical reproduction of the manuscript tables. A complete independent rerun from public data has not been verified.
