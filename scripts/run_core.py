#!/usr/bin/env python3
"""Run participant evaluations and combine completed results."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]
SRC=REPO/'src'
ALL_PIDS=tuple(f'VP{i:03d}' for i in range(1,27))
BASE_DIR='95_N4_FUNCTIONAL_ABC_LBR_FULLY_NESTED_10SEEDS_V1'

def call(argv):
 print('Running:', ' '.join(str(a) for a in argv),flush=True)
 subprocess.run([str(x) for x in argv],check=True)

def prepare_initial_bank(shin_root:Path):
 target=shin_root/BASE_DIR/'seed_01'/'fresh_cache'
 if all((target/f'{p}.npz').is_file() for p in ALL_PIDS):
  print('Seed-1 bank exists; it will not be regenerated:', target)
  return
 path=SRC/'remaining.py'
 spec=importlib.util.spec_from_file_location('hemoboundary_bank_constructor',path)
 if spec is None or spec.loader is None:raise RuntimeError('Source module unavailable')
 module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module
 spec.loader.exec_module(module)
 with tempfile.TemporaryDirectory(prefix='hemoboundary_seed1_source_') as td:
  td=Path(td)
  ref=module.import_embedded(module.REFINED_ENGINE_SOURCE,'_hb_public_refined_engine',td)
  n4=module.import_embedded(module.N4_ENGINE_SOURCE,'_hb_public_n4_engine',td)
  module.configure_repeat_seed(ref,n4,1)
  files=ref.participant_files(shin_root)
  if sorted(files)!=list(ALL_PIDS):
   raise RuntimeError(f'Requires exactly 26 Shin participants, found {sorted(files)}')
  ref.prepare_all_caches(files,target,mode='B')
 missing=[p for p in ALL_PIDS if not (target/f'{p}.npz').is_file()]
 if missing:raise RuntimeError(f'Incomplete reconstructed seed-1 bank: {missing}')
 print('Generated seed-1 bank:',target)
 print('NOTE: Numerical equivalence has not been independently verified.')

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--shin-root',required=False,default=os.environ.get('HEMOBOUNDARY_SHIN_ROOT'))
 p.add_argument('--prepare-initial-bank',action='store_true',help='Generate the missing seed-1 source test bank, never overwrite all-complete banks')
 p.add_argument('--smoke-participants',type=int,default=0,help='Run only first N participants per seed; does not complete the experiment')
 p.add_argument('--self-test-only',action='store_true')
 p.add_argument('--merge-only',action='store_true')
 args=p.parse_args(argv)
 if args.self_test_only:
  for name in ('initial.py','remaining.py'):
   call([sys.executable,SRC/name,'--self-test-only'])
  return
 if not args.shin_root: p.error('--shin-root or HEMOBOUNDARY_SHIN_ROOT is required')
 root=Path(args.shin_root).expanduser().resolve()
 if not root.is_dir():raise FileNotFoundError(f'Shin project root does not exist: {root}')
 if args.merge_only:
  call([sys.executable,SRC/'merge.py','--shin-root',root])
  return
 if args.prepare_initial_bank:prepare_initial_bank(root)
 initial_cache=root/BASE_DIR/'seed_01'/'fresh_cache'
 if not all((initial_cache/f'{pid}.npz').exists() for pid in ALL_PIDS):
  raise FileNotFoundError('Seed-1 bank missing. Use --prepare-initial-bank to generate it or supply the original saved source bank: '+str(initial_cache))
 initial=[sys.executable,SRC/'initial.py','--shin-root',root]
 remaining=[sys.executable,SRC/'remaining.py','--shin-root',root]
 if args.smoke_participants:
  if not 1<=args.smoke_participants<=26:raise ValueError('--smoke-participants must be 1..26')
  for command in (initial,remaining):command+=['--max-participants',str(args.smoke_participants)]
 call(initial)
 call(remaining)
 if args.smoke_participants:
  print('Partial smoke run only; final ten-seed merge intentionally skipped.')
 else:
  call([sys.executable,SRC/'merge.py','--shin-root',root])

if __name__=='__main__':main()
