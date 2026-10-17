#!/usr/bin/env python3
"""Check the repository structure, code, and source file hashes."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "preprocessing": {"01_SHIN_Preprocessing.ipynb", "02_Balint_Preprocessing.ipynb", "03_Khan_Preprocessing.ipynb"},
    "analysis": {"01_HemoBoundary_Main.ipynb", "02_Component_Ablation.ipynb", "03_Functional_Necessity.ipynb", "04_Seven_Method_Benchmark.ipynb", "05_Unseen_Morphology_Robustness.ipynb", "06_Balint_Khan_External_Validation.ipynb", "07_Late_Window_Recovery.ipynb"},
}


def main():
    folders = {p.name for p in ROOT.iterdir() if p.is_dir()}
    assert folders == {"preprocessing", "analysis", "src", "scripts", "docs"}, folders
    assert not (ROOT / 'src' / 'hemoboundary').exists(), 'Unexpected extra source package directory'
    for name in ('initial.py', 'remaining.py', 'merge.py', 'benchmark.py', 'external.py'):
        assert (ROOT / 'src' / name).is_file(), f'Missing source file: {name}'
    for folder, expected in EXPECTED.items():
        names = {p.name for p in (ROOT / folder).glob("*.ipynb")}
        assert names == expected, (folder, names ^ expected)

    for path in ROOT.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert cells
        for cell in cells:
            assert not cell.get("outputs") and cell.get("execution_count") is None
            src = cell["source"]
            ast.parse("".join(src) if isinstance(src, list) else src)

    for path in ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert not re.search(r"(?m)^\s*#\s*%%", source), path

    manifest = json.loads((ROOT / "docs" / "SOURCE_HASHES.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["sha256"].items():
        path = ROOT / name
        assert path.is_file(), name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected_hash, name

    blocked = {".mat", ".npz", ".h5", ".hdf5", ".snirf", ".csv", ".gz", ".zip", ".tar"}
    source_archive = ROOT / "docs" / "provenance" / "frozen_comparison_source.zip"
    assert hashlib.sha256(source_archive.read_bytes()).hexdigest() == "a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d"
    assert not [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in blocked and p != source_archive]
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".ipynb", ".md"}:
            assert ("6298861b" + "1dc6be936d73") not in path.read_text(encoding="utf-8")

    print("PASS: repository files, notebook cells, Python syntax, hashes and data exclusions")


if __name__ == "__main__":
    main()
