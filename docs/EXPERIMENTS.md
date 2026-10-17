# Experiments

| Analysis notebook | Source implementation | Evaluation |
|---|---|---|
| Main evaluation | `initial.py`, `remaining.py`, `engines/` | Participant-level nested evaluation, 10 seeds |
| Component ablation | `merge.py` | Cumulative components and participant summaries |
| Functional necessity | `functional.py`, `functional_merge.py` | Component neutralization |
| Seven-method benchmark | `benchmark.py` | Reference timing comparisons |
| Morphology robustness | `morphology.py`, `morphology_merge.py` | Alternative response shapes |
| External validation | `external.py` | Bálint and Khan |
| Late-window recovery | `late_recovery_shin_balint.py`, `late_recovery_khan.py` | Recorded task responses |

The eight additional comparator adaptations and their sensitivity tests are supported under `src/expanded/`; see [expanded comparison](EXPANDED_COMPARISON.md).
