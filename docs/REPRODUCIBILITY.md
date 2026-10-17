# Reproducibility status

The repository includes the HemoBoundary core model, seven original reference comparisons, the exact frozen expanded-comparison source, the two Cohen-Adad GLM adaptations, the two Lina wavelet adaptations, participant-level expanded-comparison statistics, and missing-estimate sensitivity scripts.

The original frozen source archive and separate published-method adaptation scripts have verified source hashes. Available synthetic software self-tests and code checks are not equivalent to a complete numerical rerun using the public datasets.

The archived expanded-comparison statistics and sensitivity routines expect particular historical checkpoint trees and a verified 770-case export. Neither those generated results nor the recorded participant datasets are included, and a new benchmark output tree is not necessarily interchangeable with the historical export. End-to-end recreation of the manuscript's reported numbers remains to be demonstrated from independently downloaded data.

See [expanded comparison](EXPANDED_COMPARISON.md) and [run order](RUN_ORDER.md) for source verification and reproduction steps.
