# Study analyses

| Analysis | Implementation | Evaluation |
|---|---|---|
| Shin source evaluation | `src/initial.py`, `src/remaining.py`, `src/engines/` | 26 participant-held-out folds and 10 source seeds |
| Component ablation | `src/merge.py` | Cumulative contribution of each component |
| Component perturbation | `src/functional.py`, `src/functional_merge.py` | Components disabled within the fitted model without retraining |
| Seven-method benchmark | `src/benchmark.py` | Timing methods with a shared response gate |
| Response morphology | `src/morphology.py`, `src/morphology_merge.py` | Evaluation on unfamiliar generated response shapes |
| Bálint and Khan transfer | `src/external.py` | Known-boundary response banks without target-label adaptation |
| Late recovery | `src/late_recovery_shin_balint.py`, `src/late_recovery_khan.py` | Descriptive analysis of recorded task responses |
| Recorded resting backgrounds | `src/recorded_background.py`, `src/reproduction_utils.py` | 14 participants, 224 conditions, single-source comparisons and 10-source robustness |
| Nested classical comparisons | `src/nested_classical.py` | Eight Ridge and ExtraTrees configurations, 26 outer participants, 10 seeds |
| Transfer diagnostics | `src/transfer_diagnostics.py` | Response probabilities, threshold check, and paired AR background control |

The expanded comparison code for the three primary datasets is in `src/expanded/`. The recorded-background adaptations of Cohen-Adad, Lina, and HRfunc are in `src/recorded_cohen.py`, `src/recorded_lina.py`, and `src/recorded_hrfunc.py`. The latter require source-calibration records and reference files that are not distributed with the repository.

The published-method comparisons on recorded backgrounds use one source configuration. The ten-seed robustness experiment repeats only HemoBoundary. Results from these experiments are not combined into a new paired comparison.

Known timing targets were generated for the benchmark experiments. Timing accuracy against independently annotated naturally occurring fNIRS responses remains untested.
