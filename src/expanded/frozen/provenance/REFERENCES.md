# Primary sources consulted

Pinti et al. (2017). A novel GLM-based method for the Automatic IDentification of
functional Events (AIDE) in fNIRS data recorded in naturalistic environments.
NeuroImage 155, 291-304. DOI: 10.1016/j.neuroimage.2017.05.001.
Official repository: https://github.com/multimodalspectroscopy/AIDE
The V8 implementation is explicitly a single-event search-core adaptation, not
a full reproduction of that repository.

Santosa et al. (2019). Investigation of the sensitivity-specificity of canonical-
and deconvolution-based linear models in evoked functional near-infrared
spectroscopy. Neurophotonics 6(2), 025009. DOI: 10.1117/1.NPh.6.2.025009.
https://pmc.ncbi.nlm.nih.gov/articles/PMC6541797/
Coefficient algorithms inspected:
https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmath/ar_irls.m
https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmath/ar_fit.m
https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmath/stepwise.m
https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmath/lagmatrix.m
https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmath/infocrit.m
Only independent coefficient-level reimplementations are supplied. The full
MATLAB statistical inference, runtime and paper evaluation are not reproduced.

Nilearn 0.12.1 SPM basis reference:
https://github.com/nilearn/nilearn/blob/0.12.1/nilearn/glm/first_level/hemodynamic_models.py
The installed package is used for numerical basis verification at runtime.

Schaedig et al. (2026 volume; online publication 2025). HRfunc: a tool for modeling
hemodynamic response variability in fNIRS. Neurophotonics 13(S1), S17801.
DOI: 10.1117/1.NPh.13.S1.S17801.
https://github.com/dennys246/HRfunc
https://pypi.org/project/hrfunc/1.3.1/
V8 pins the author package distribution to 1.3.1 and the wheel SHA256 in
requirements.txt. Repository main was inspected for algorithm/API understanding;
no claim is made that the current main branch equals an unpinned historical
release. Runtime algebra checks guard the installed estimator behavior.

Maheux et al. (2017). Optimal measurements of hemodynamic response latency in fNIRS
using the jackknife approach. Psychophysiology 54(1), 62-73.
DOI: 10.1111/psyp.12648. https://pubmed.ncbi.nlm.nih.gov/28000262/
Percentage-of-peak is the cited measure family. The selected 20%/50% single-trial
benchmarks do not reproduce its jackknife statistical procedure.

Original HemoBoundary generation and external loading functions came from the
user-provided HEMOBOUNDARY_OPTIONB_TABLE3_10SEED_BALINT_KHAN_ZERO_SHOT_COMPLETE.py.
DATA_SOURCE_MANIFEST.json records the recovered original function hashes. Only
loading, profiling and generation code is used; no HemoBoundary training routine
is executed.

Barker, Aarabi and Huppert (2013). Autoregressive model based algorithm for
correcting motion and serially correlated errors in fNIRS. Biomedical Optics
Express 4(8), 1366-1379. DOI: 10.1364/BOE.4.001366.
https://pubmed.ncbi.nlm.nih.gov/24009999/
This is the methodological source for AR-IRLS, distinct from Santosa's later
evaluation of response basis choices.
