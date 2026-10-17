# Hierarchical configuration selection

Candidate settings were selected using three participant-grouped inner folds from the source participants. Each stage inherited the settings selected at the preceding stage.

| Stage | Candidate settings | Count |
|---|---|---:|
| A | `(state_C, timing_alpha)`: `(0.01, 10)`, `(0.003, 10)`, `(0.03, 10)`, `(0.01, 3)`, `(0.01, 30)` | 5 |
| B | `change_frac`: `0.20`, `0.30`, `0.40` | 3 |
| C | `response_threshold`: `0.40`, `0.50`, `0.60` | 3 |
| LBR | `(radius, local_alpha, strength)`: `(1, 100, .50)`, `(.5, 100, .50)`, `(1.5, 100, .50)`, `(1, 30, .50)`, `(1, 300, .50)`, `(1, 100, .25)`, `(1, 100, .75)` | 7 |

Every stage minimizes the same source-validation loss, combining onset, peak, offset/1.5, and duration/1.5 losses. Missing eligible estimates receive a 4-s penalty, while finite errors are not capped. Reference-censored offset and duration targets are excluded. Candidate order resolves ties.

These are stage-specific grids, not a full factorial search. Held-out target participants are excluded from configuration selection.
