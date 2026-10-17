#!/usr/bin/env python3
"""Run archived comparison analyses without modifying scientific source files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / 'src' / 'expanded'
ORIGINAL = ROOT / 'docs' / 'provenance' / 'frozen_comparison_source.zip'
FROZEN = EXPANDED / 'frozen'
EXPECTED = 'a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d'
ADAPTERS = {
    'cohen': ('cohen_adad.py', 'COHENADAD2007_FROZEN_V8_ARCHIVE_a953339593ea.zip'),
    'lina': ('lina_wavelet.py', 'LINA2010_FROZEN_V8_ARCHIVE_a953339593ea.zip'),
}

def check_sources():
    data = ORIGINAL.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED, 'Frozen archive checksum mismatch'
    with zipfile.ZipFile(ORIGINAL) as archive:
        manifest = json.loads(archive.read('PACKAGE_FILES_SHA256.json'))
        for name, expected in manifest.items():
            if name == 'README.md':
                source = archive.read(name)
            else:
                candidate = FROZEN / name
                assert candidate.is_file(), name
                source = candidate.read_bytes()
            assert hashlib.sha256(source).hexdigest() == expected, name
    fingerprints = {
        'cohen_adad.py': 'f238bbc389faca02cc3a67452805d6b69b0c932b1abb30734759af26b596998c',
        'lina_wavelet.py': '0b2b4b65312246acacf72e9cdc230b35851f010819e104f7fc3ff463ff99b59d',
    }
    for name, expected in fingerprints.items():
        assert hashlib.sha256((EXPANDED / name).read_bytes()).hexdigest() == expected, name
    print('PASS: exact frozen comparison and two original adaptation sources')

def stage_archive(target_name):
    target = Path('/content') / target_name
    if not target.parent.is_dir():
        raise RuntimeError('This archived runner expects Google Colab and /content.')
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED:
            raise RuntimeError(f'Existing archive differs from verified source: {target}')
    else:
        shutil.copyfile(ORIGINAL, target)
    return target

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('check', 'frozen', 'cohen', 'lina', 'paired', 'sensitivity', 'review'))
    p.add_argument('--mode', default='selftest', help='For frozen: preflight/pilot/seed1/full/summarize; for adapters: selftest/pilot/full/summary')
    p.add_argument('--max-new-cases', type=int, default=0)
    p.add_argument('--root', default=None, help='Original dataset project directory; supported by frozen benchmark only')
    p.add_argument('--output', default=None, help='New frozen benchmark output directory')
    args = p.parse_args()
    check_sources()
    if args.stage == 'check':
        return
    if args.stage == 'frozen':
        mode = 'preflight' if args.mode == 'selftest' else args.mode
        cmd = [sys.executable, str(FROZEN / 'run_colab.py'), '--mode', mode]
        if args.root:
            cmd += ['--root', args.root]
        if args.output:
            cmd += ['--output', args.output]
    elif args.stage in ADAPTERS:
        file_name, archive_name = ADAPTERS[args.stage]
        if args.mode != 'selftest':
            stage_archive(archive_name)
        cmd = [sys.executable, str(EXPANDED / file_name), '--mode', args.mode]
        if args.max_new_cases:
            cmd += ['--max-new-cases', str(args.max_new_cases)]
    elif args.stage in ('paired', 'sensitivity'):
        name = 'paired_statistics.py' if args.stage == 'paired' else 'sensitivity.py'
        cmd = [sys.executable, str(EXPANDED / name)]
        if args.mode == 'selftest':
            cmd += ['--selftest']
    else:
        cmd = [sys.executable, str(EXPANDED / 'sensitivity_review.py')]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
