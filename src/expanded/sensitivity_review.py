#!/usr/bin/env python3
"""Verify directions of saved HemoBoundary V8 sensitivity contrasts.

Reads ONLY prior completed results. Does NOT rerun models, predictions, bootstrap
or sign-flip testing. Designed for Colab with Drive mounted at /content/gdrive.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

RUN = Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1/HB_V8_THREE_CHECKS_READONLY_NEW_ONLY/V8_770_THREE_CHECKS_20261016T094735579816Z')
FAMILIES = ('penalty_2s', 'penalty_4s', 'penalty_6s',
            'conditional_mae_valid_only', 'common_valid_mae', 'coverage')
DATASETS = ('SHIN', 'BALINT', 'KHAN')
PIDS = {'SHIN': 26, 'BALINT': 26, 'KHAN': 25}
BOUNDARIES = ('onset', 'peak', 'offset', 'duration')
N_PER_DATASET = 32 # eight comparators x four boundaries
EPS = 1e-10


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def run_check(run=RUN):
    audit_file = run/'AUDIT_COMPLETED.json'
    results_file = run/'PAIRED_ALL_SIX_FAMILIES.csv'
    if not audit_file.is_file() or not results_file.is_file():
        raise FileNotFoundError('Saved completed results not found in: '+str(run))
    audit=json.loads(audit_file.read_text(encoding='utf-8'))
    assert audit['status']=='ALL_770_READONLY_SAVED_PREDICTIONS_REPLAY_VERIFIED', 'Original run not verified'
    assert audit['original_770_cases']==770 and audit['saved_4s_case_scores_reproduced'], '770/770 audit incomplete'
    assert audit['prior_4s_paired_parity']['paired_comparisons_verified']==96, 'Original 4s paired parity not proven'
    assert audit['output_sha256']['PAIRED_ALL_SIX_FAMILIES.csv']==sha256(results_file), 'CSV checksum mismatch'
    assert audit['family_sizes']=={f:96 for f in FAMILIES}, 'Expected 96 per family'
    assert audit['participants_by_dataset']==PIDS, 'Dataset participant counts mismatch'

    with results_file.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==576, f'Expected 576 paired rows, got {len(rows)}'
    seen=set()
    by=defaultdict(list)
    adverse=[]
    for r in rows:
        f,d,m,b=r['family'],r['dataset'],r['method'],r['boundary']
        key=(f,d,m,b)
        assert key not in seen, f'Duplicate key: {key}'
        seen.add(key)
        assert f in FAMILIES and d in DATASETS and b in BOUNDARIES, key
        assert r['inference_status']=='COMPLETE', key
        assert int(r['n_participants_valid'])==PIDS[d], key
        assert int(r['n_participants_expected'])==PIDS[d], key
        assert int(r['n_source_seeds'])==10 and int(r['holm_family_size'])==96, key
        gain,p=float(r['mean_gain']),float(r['holm_p'])
        assert math.isfinite(gain) and math.isfinite(p) and 0<=p<=1, key
        significant=p<0.05
        assert r['holm_significant'].lower() == str(significant).lower(), key
        # From original V8 analysis: gain > 0 ALWAYS means better HemoBoundary.
        # For losses, gain=competitor-HB; for coverage, gain=HB-competitor.
        r['gain']=gain
        r['significant']=significant
        by[(f,d)].append(r)
        if significant and gain<=EPS:
            adverse.append(r)

    print('VERIFIED: source audit, 770-case completion, 96 original 4s comparisons and CSV SHA256')
    print('Mean gain > 0 favors HemoBoundary for every metric (including coverage).')
    print('\nFamily                            Positive/96  Negative/96  Tie/96  Significant pro-HB/96  Significant pro-comparator/96')
    five_all=True
    for f in FAMILIES:
        subset=sum((by[(f,d)] for d in DATASETS),[])
        assert len(subset)==96, f'Incomplete family {f}'
        assert all(len(by[(f,d)])==32 for d in DATASETS), f'Incomplete dataset in {f}'
        pos=sum(r['gain']>EPS for r in subset)
        neg=sum(r['gain']<-EPS for r in subset)
        sig_pos=sum(r['significant'] and r['gain']>EPS for r in subset)
        sig_neg=sum(r['significant'] and r['gain']<-EPS for r in subset)
        tied=96-pos-neg
        print(f'{f:34s}{pos:>5d}       {neg:>5d}    {tied:>5d}             {sig_pos:>5d}                       {sig_neg:>5d}')
        if f!='coverage' and (pos!=96 or sig_pos!=96 or sig_neg!=0):five_all=False

    print('\nSignificant differences that DO NOT favor HemoBoundary:')
    if adverse:
        for r in adverse:
            print(f'  {r["family"]:28s} {r["dataset"]:6s} {r["display_method"]:32.32s} {r["boundary"]:8s} mean_gain={r["gain"]:+.6f} Holm p={float(r["holm_p"]):.5g}')
    else:
        print('  NONE')

    if five_all:
        print('\nPASS: All 96 paired mean gains favor HemoBoundary and are Holm-significant in EACH of the five timing/error families (2/4/6s, conditional, common-valid).')
    else:
        print('\nCAUTION: Do not claim that all 96 timing/error contrasts favor HemoBoundary. See directions above.')
    if not adverse:
        print('PASS: EVERY Holm-significant difference across all six families favors HemoBoundary.')
    else:
        print('CAUTION: Some Holm-significant differences favor a comparator or are near zero.')
    return rows


if __name__=='__main__':
    try:
        from google.colab import drive
        if not Path('/content/gdrive/MyDrive').is_dir():
            drive.mount('/content/gdrive')
    except ImportError:
        pass
    run_check()
