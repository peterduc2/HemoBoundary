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

## Source-nested classical controls

A separate Shin comparison used the same 26 held-out participants and ten original source seeds. It evaluated Ridge and ExtraTrees against four representations: raw, state-only, multiscale-only, and combined state and multiscale features. The state representation was fitted within each inner training fold.

| Selected parameter | Candidate values |
|---|---|
| Ridge alpha | 1, 3, 10, 30, 100 |
| ExtraTrees minimum leaf size | 1, 4, 8 |
| Response threshold | 0.20, 0.40, 0.50, 0.60 |
| State encoder C | Fixed at 0.01 for this diagnostic control |
| Multiscale change fraction | Fixed at 0.30 |
| Detector regularization C | Fixed at 0.5 |

Three participant-grouped inner folds selected the readout and threshold. Source training banks were used for fitting, while the held-out source participants' separate test banks were used for inner evaluation. The outer test participant was excluded throughout selection.

The eight classical configurations produced 32 timing comparisons for each reporting rule: native gate, shared HemoBoundary gate, and ungated positive responses. Holm correction was applied separately within those three 32-test families. The eight independent detection contrasts formed a separate family. Results were averaged over the ten source seeds within each of the 26 outer participants before paired inference. The tested search did not include all possible model architectures.
