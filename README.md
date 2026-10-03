# HemoBoundary

HemoBoundary is a framework for trial-level temporal boundary estimation in functional near-infrared spectroscopy (fNIRS).

The framework estimates the onset, primary peak, and recovery of individual hemodynamic responses while allowing recovery to remain unresolved when it is not observed within the available recording window.

This repository contains the preprocessing and analysis code used for the manuscript:

**HemoBoundary: State Calibrated Fusion for Trial-Level fNIRS Temporal Boundary Estimation**

## Repository structure

```text
HemoBoundary/
├── preprocessing/
│   ├── 01_SHIN_Preprocessing.ipynb
│   ├── 02_Balint_Preprocessing.ipynb
│   └── 03_Khan_Preprocessing.ipynb
│
├── analysis/
│   ├── 01_HemoBoundary_Main.ipynb
│   ├── 02_Component_Ablation.ipynb
│   ├── 03_Functional_Necessity.ipynb
│   ├── 04_Seven_Method_Benchmark.ipynb
│   ├── 05_Unseen_Morphology_Robustness.ipynb
│   ├── 06_Balint_Khan_External_Validation.ipynb
│   └── 07_Late_Window_Recovery.ipynb
│
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```
