"""Run the preserved fourth-dataset published-method adaptations."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {'cohen': 'src.recorded_cohen', 'lina': 'src.recorded_lina',
           'hrfunc': 'src.recorded_hrfunc'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=(*MODULES, 'all'))
    p.add_argument('--project-root', type=Path, required=True,
                   help='Parent directory containing original Shin and saved source-only calibration artifacts')
    p.add_argument('--output', type=Path, required=True,
                   help='Folder outside the repository for generated adapters output')
    args = p.parse_args()
    env = dict(os.environ, HEMOBOUNDARY_PROJECT_ROOT=str(args.project_root.expanduser().resolve()),
               HEMOBOUNDARY_RECORDED_ADAPTER_OUTPUT=str(args.output.expanduser().resolve()))
    stages = list(MODULES) if args.family == 'all' else [args.family]
    for family in stages:
        subprocess.run([sys.executable, '-m', MODULES[family]], check=True,
                       cwd=ROOT, env=env)


if __name__ == '__main__':
    main()
