# Expanded-comparison source

This folder contains the scientific implementation and software tests for the expanded comparison. The model, evaluation, and test modules are preserved with their source hashes. A source snapshot is provided in `docs/provenance/frozen_comparison_source.zip` at the repository root.

## Verification and execution

See [expanded comparison instructions](../../../docs/EXPANDED_COMPARISON.md) for dataset requirements, source verification, and run order.

```bash
python scripts/run_expanded_comparison.py check
python scripts/run_expanded_comparison.py frozen --mode preflight
python scripts/run_expanded_comparison.py frozen --mode full
```

The remaining comparison and statistical stages are described in the same guide. Full execution requires the prepared datasets and output files from the preceding analyses. Source checks and software tests do not establish an independent numerical reproduction of the manuscript results.
