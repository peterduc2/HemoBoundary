#!/usr/bin/env python3
"""Validate the original frozen expanded-comparison source archive SHA-256 only."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
EXPECTED='a953339593ea228ef36aa618e5f2b0075c9153830b91c6daf50eef0e7e354d5d'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    a=parser.parse_args()
    if not a.archive.is_file():
        parser.error(f'Archive not found: {a.archive}')
    actual=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    print('Recorded SHA-256:',EXPECTED)
    print('Observed SHA-256:',actual)
    if actual!=EXPECTED:
        raise SystemExit('FAIL: this is not the original frozen expanded-comparison archive.')
    print('PASS: frozen archive byte identity. Numerical reproduction is NOT established.')

if __name__=='__main__':main()
