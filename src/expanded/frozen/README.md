# Expanded comparison: scientific source

This folder provides the original scientific implementation of the expanded biomedical-signal comparison. The executable Python modules, tests, and configuration files are preserved from the verified source archive. The **unaltered original package**, including its original README, is available at `docs/provenance/frozen_comparison_source.zip` from the repository root.

## Use this repository's instructions

See [Expanded comparison](../../../../docs/EXPANDED_COMPARISON.md) for the current source-file map, dataset and checkpoint requirements, and complete run sequence. Start with the source-verification command from the repository root:

```bash
python scripts/run_expanded_comparison.py check
```

Then perform a source preflight before the full benchmark:

```bash
python scripts/run_expanded_comparison.py frozen --mode preflight
python scripts/run_expanded_comparison.py frozen --mode full
```

Use the `cohen`, `lina`, `paired`, `sensitivity`, and `review` stages documented in the same guide for the remaining analyses. Full execution needs external input data and saved experimental results. The source checks alone do not establish numerical reproduction.

The documentation in this README is specific to the public repository. The unmodified archive remains the record of the source implementation and its original execution environment.
