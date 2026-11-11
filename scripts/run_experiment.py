#!/usr/bin/env python3
"""Dispatch the unified experiments without duplicating scientific methods."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1]
SRC=REPO/'src'

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('experiment',choices=['functional','morphology','benchmark','external','late-shin-balint','late-khan'])
 p.add_argument('--shin-root',default=os.environ.get('HEMOBOUNDARY_SHIN_ROOT',''))
 p.add_argument('--balint-h5',default='')
 p.add_argument('--khan-h5',default='')
 p.add_argument('--self-test-only',action='store_true')
 p.add_argument('--seeds',default='',help='Subset only for benchmark/external testing; empty uses all 10')
 p.add_argument('--skip-merge',action='store_true')
 args=p.parse_args(argv)
 if args.shin_root:
  os.environ['HEMOBOUNDARY_SHIN_ROOT']=str(Path(args.shin_root).expanduser().resolve())
 jobs={
  'functional':('functional.py','functional_merge.py'),
  'morphology':('morphology.py','morphology_merge.py'),
  'benchmark':('benchmark.py',),
  'external':('external.py',),
  'late-shin-balint':('late_recovery_shin_balint.py',),
  'late-khan':('late_recovery_khan.py',),
 }
 selected=jobs[args.experiment]
 for index,name in enumerate(selected):
  if index>0 and (args.skip_merge or args.self_test_only):break
  command=[sys.executable,str(SRC/name)]
  if args.experiment in ('benchmark','external'):
   if args.shin_root:command+=['--shin-root',str(Path(args.shin_root).expanduser().resolve())]
   if args.seeds:command+=['--seeds',args.seeds]
  if args.experiment=='external':
   if args.balint_h5:command+=['--balint-h5',args.balint_h5]
   if args.khan_h5:command+=['--khan-h5',args.khan_h5]
  if args.experiment=='late-shin-balint':
   if args.shin_root:command+=['--shin-root',args.shin_root]
   if args.balint_h5:command+=['--balint-h5',args.balint_h5]
  if args.experiment=='late-khan' and args.khan_h5:command+=['--h5',args.khan_h5]
  if args.self_test_only and args.experiment in ('benchmark','external','late-shin-balint','late-khan'):
   command.append('--self-test-only')
  if args.self_test_only and args.experiment in ('functional','morphology'):
   print('For this experiment, run scripts/audit_structure.py.');return
  print('Running',name,flush=True)
  subprocess.run(command,check=True,env=os.environ.copy())

if __name__=='__main__':main()
