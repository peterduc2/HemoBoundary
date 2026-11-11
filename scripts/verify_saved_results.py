#!/usr/bin/env python3
"""Compare saved experiment summaries with recorded values."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

TOL = 5e-6

CORE = {
 "Base": (0.992147,0.998478,1.071618,1.977043,2.663602),
 "Base + A": (0.992147,0.287081,0.293875,1.977722,2.036883),
 "Base + A + B": (0.992147,0.199152,0.196894,0.454228,0.482575),
 "Base + A + B + C": (0.998798,0.166768,0.165416,0.425741,0.454547),
 "HemoBoundary": (0.998798,0.129562,0.132547,0.384489,0.406216),
}
REFERENCE = {
 "R1_THRESHOLD20": (0.870230,0.986897,1.392649,2.198923),
 "R2_THRESHOLD50": (1.380835,0.986897,2.644693,4.017678),
 "R3_FRACTIONAL_AREA": (1.654773,1.891706,2.934322,4.165151),
 "R4_FLEXIBLE_SPLINE": (0.521819,1.114875,1.472087,1.916425),
 "R5_CHANGE_POINT": (1.315370,0.988259,2.480374,3.786419),
 "R6_ORDERED_STATE": (0.350391,0.967467,0.672421,0.896831),
 "R7_CANONICAL_HRF_SHIFTED_GLM": (0.725445,1.338822,2.049794,2.501134),
}
EXTERNAL = {
 ("BALINT","OPTIONB_ABC"): (0.994631410,0.242459510,0.243927399,0.633937808,0.683075149),
 ("BALINT","OPTIONB_HEMOBOUNDARY"): (0.994631410,0.212044809,0.218562434,0.614194115,0.648674102),
 ("KHAN","OPTIONB_ABC"): (0.983750000,0.284432410,0.291462669,0.688255871,0.718969736),
 ("KHAN","OPTIONB_HEMOBOUNDARY"): (0.983750000,0.254878305,0.263332124,0.665102429,0.697058307),
}
COLS = ["response_balanced_accuracy_mean","onset_mae_sec_mean","peak_mae_sec_mean","offset_mae_sec_mean","duration_mae_sec_mean"]
TIMING_COLS = COLS[1:]

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing results file: {path}")
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def assert_values(tag, observed, expected, tol=TOL):
    for k,got,want in zip(["BA","onset","peak","offset","duration"][-len(expected):],observed,expected):
        if not math.isclose(got,want,rel_tol=0,abs_tol=tol):
            raise AssertionError(f"{tag}: {k} observed={got} expected={want}, tol={tol}")
    print(f"PASS {tag}")

def check_main(root):
    path = root / '100_B0_STRICT_HIERARCHICAL_FINAL_10SEED_MERGE_V1' / 'T2_10SEED_MEAN_SD_ABLATION.csv'
    rows = read_csv(path)
    if len(rows) != 5:
        raise AssertionError(f"main evaluation expects 5 rows, got {len(rows)}")
    found={}
    for row in rows:
        label=row['configuration'].strip()
        if 'LBR' in label or 'HemoBoundary' in label:
            label='HemoBoundary'
        if label in found: raise AssertionError(f"duplicate main evaluation row {label}")
        found[label]=row
    if set(found)!=set(CORE): raise AssertionError(f"main evaluation row labels mismatch: {set(found)^set(CORE)}")
    for label,expected in CORE.items():
        obs=tuple(float(found[label][c]) for c in COLS)
        assert_values('main evaluation '+label,obs,expected)

def check_benchmarks(root):
    path=root / '101_OPTIONB_TABLE2_10SEED_BENCHMARK_V1' / 'tables' / 'Table2_OPTIONB_10SEED_BENCHMARK_MEAN_SD.csv'
    rows=read_csv(path)
    found={row['method']:row for row in rows}
    for label,expected in REFERENCE.items():
        if label not in found: raise AssertionError(f"timing benchmarks missing method {label}")
        obs=tuple(float(found[label][c]) for c in TIMING_COLS)
        for k,got,want in zip(['onset','peak','offset','duration'],obs,expected):
            if not math.isclose(got,want,rel_tol=0,abs_tol=TOL):
                raise AssertionError(f"timing benchmarks {label} {k}: observed={got} expected={want}")
        print(f"PASS timing benchmarks {label}")

def check_external(root):
    path=root / '102_OPTIONB_TABLE3_10SEED_BALINT_KHAN_ZERO_SHOT_V1' / 'tables' / 'Table3_OPTIONB_10SEED_EXTERNAL_MEAN_SD.csv'
    rows=read_csv(path)
    found={(r['dataset'].upper(),r['model']):r for r in rows}
    if set(found)!=set(EXTERNAL):
        raise AssertionError(f"external evaluation row keys mismatch: {set(found)^set(EXTERNAL)}")
    for key,expected in EXTERNAL.items():
        obs=tuple(float(found[key][c]) for c in COLS)
        assert_values('external evaluation '+str(key),obs,expected,tol=5e-7)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,required=True,
        help='Directory containing saved experiment results')
    a=p.parse_args()
    root=a.project_root.expanduser().resolve()
    errors=[]
    for ver,fn in (("main evaluation",check_main),("timing benchmarks",check_benchmarks),("external evaluation",check_external)):
        try: fn(root)
        except (AssertionError, FileNotFoundError, KeyError,ValueError) as e:
            print(f"FAIL {ver}: {e}")
            errors.append(ver)
    if errors:
        print('NOT VERIFIED:',', '.join(errors))
        raise SystemExit(1)
    print('PASS: Saved experiment summaries match expected values.')
    print('This check does not rerun models or verify the expanded comparison.')

if __name__=='__main__': main()
