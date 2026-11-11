# Code organization

Scientific Python modules are located directly in `src/`. The existing `engines/` and `expanded/` modules provide model components and comparator implementations. The `preprocessing/` folder contains four dataset preparation notebooks, while `analysis/` contains ten notebooks corresponding to the main experiments and additional evaluations. The `scripts/` folder contains command-line entry points and validation checks.

Random-number generation contexts and feature-construction rules are retained in the scientific implementation because they affect fitted models and reproducibility. Dataset files and generated results are stored outside the repository.
