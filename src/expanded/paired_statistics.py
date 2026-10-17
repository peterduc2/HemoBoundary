#!/usr/bin/env python3
"""HemoBoundary BSPC Table 4: READ-ONLY participant-paired inference (2026-10-16).

Inputs (EXACT paths within the previously confirmed HemoBoundary Google Drive root):
  * V8 V1/V2 770 SHA-verified derived export (long-form participant-level data)
  * Cohen-Adad 2007 independent paper-guided trial adaptation, 770 verified cases
  * Lina 2010 independent paper-guided trial adaptation, 770 verified cases

No training, no fitting, no editing source files, no republishing native author SOTA.
Everything written to a NEW results-only directory. Option: --selftest (no Drive).

Family-wise Holm adjustment across 96 timing and 24 detection tests respectively.
Positive differences favor HemoBoundary, and confidence intervals are percentile
participant-bootstrap CIs (not family-wise simultaneous confidence intervals).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/content/gdrive/MyDrive/HonorProject/shin2018_word_generation_state_preparation_v1')
V8_EXPORTED = ROOT / 'HB_V8_V1_V2_VERIFIED_770_EXPORT_20261015T041547Z_6190f376'
V8_V1 = ROOT / 'HB_V8_MULTICHANNEL_PUBLISHED_COMPARISON'
V8_V2 = ROOT / 'HB_V8_CPU_ONLY_SEEDS_06_TO_10'
COHEN = ROOT / 'HB_COHENADAD2007_SHIFT_GLM_PAPER_GUIDED_770_V1'
LINA = ROOT / 'HB_LINA2010_WAVELET_HRF_PAPER_GUIDED_770_V1'
OUTPUT_PARENT = ROOT / 'HB_BSPC_TABLE4_PAIRED_STATS_NEW_ONLY'

V8_PROTO_SHA = '6190f376593bb9befc92427bf2efbf70410194215e3dd60baacf325fd3864d55'
COHEN_CODE_SHA = 'f238bbc389faca02cc3a67452805d6b69b0c932b1abb30734759af26b596998c'
LINA_CODE_SHA = '0b2b4b65312246acacf72e9cdc230b35851f010819e104f7fc3ff463ff99b59d'

DATASETS = ('SHIN', 'BALINT', 'KHAN')
PIDS = {
    'SHIN': tuple(f'VP{i:03d}' for i in range(1, 27)),
    'BALINT': tuple(f'C{i:02d}' for i in range(1, 27)),
    'KHAN': tuple(f'S{i:02d}' for i in range(1, 26)),
}
SEEDS = 10
EXPECTED_CASES = 770
HB = 'HEMOBOUNDARY_STRICT_SAVED'
# Eight *distinct baseline implementations*, not eight independent published papers.
METHODS = (
    HB,
    'EXTRATREES9_CONTROL',
    'RIDGE9_CONTROL',
    'COHENADAD2007_SHIFT_GLM_V8_ADAPTER',
    'COHENADAD2007_DERIVATIVE_GLM_V8_ADAPTER',
    'LINA2010_WAVELET_PCA1_V8_ADAPTER',
    'LINA2010_WAVELET_NOPCA_V8_ADAPTER',
    'HRFUNC_AUTHOR__NATIVE_THRESH',
    'HRFUNC_AUTHOR__SOURCE_RIDGE_READOUT',
)
DISPLAY = {
    HB: 'HemoBoundary',
    'EXTRATREES9_CONTROL': 'ExtraTrees',
    'RIDGE9_CONTROL': 'Ridge',
    'COHENADAD2007_SHIFT_GLM_V8_ADAPTER': 'Cohen-Adad shift GLM',
    'COHENADAD2007_DERIVATIVE_GLM_V8_ADAPTER': 'Cohen-Adad derivative GLM',
    'LINA2010_WAVELET_PCA1_V8_ADAPTER': 'Lina wavelet + PCA',
    'LINA2010_WAVELET_NOPCA_V8_ADAPTER': 'Lina wavelet no PCA',
    'HRFUNC_AUTHOR__NATIVE_THRESH': 'HRfunc V8 native threshold',
    'HRFUNC_AUTHOR__SOURCE_RIDGE_READOUT': 'HRfunc V8 source Ridge',
}
SOURCE = {m: ('Cohen' if m.startswith('COHEN') else 'Lina' if m.startswith('LINA') else 'V8') for m in METHODS}
TIMING = ('onset', 'peak', 'offset', 'duration')
METRICS = tuple(f'{b}_loss_missing4_sec' for b in TIMING)
BA = 'balanced_accuracy'
FAMILY_TIMING = 'timing_96_across_all_datasets_methods_boundaries'
FAMILY_BA = 'detection_24_across_all_datasets_methods'
SIGN_FLIPS = 100_000
BOOTSTRAPS = 20_000
RNG_SEED = 20261016


def fail(message):
    raise RuntimeError(message)


def load_json(p):
    if not p.is_file():
        raise FileNotFoundError(f'Required saved result is missing: {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_hash(obj):
    b = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                   allow_nan=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()


def parse_finite(v, context):
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Missing/unparseable numeric result: {context}: {v!r}') from exc
    if not math.isfinite(x):
        fail(f'Nonfinite participant metric: {context}: {x!r}')
    return x


def read_csv(p):
    if not p.is_file():
        raise FileNotFoundError(f'Required saved table is missing: {p}')
    with p.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(p, data, columns):
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        w.writerows(data)


def check_exact_subjects(table, expected, label):
    found = set(table)
    if found != expected:
        missed, extra = expected-found, found-expected
        fail(f'{label}: participants/method/metric incomplete or unexpected; '
             f'missing {len(missed)} {sorted(missed)[:4]}, extras {len(extra)} {sorted(extra)[:4]}')


def verify_v8_provenance(root=ROOT, original_export=V8_EXPORTED,
                         v1=V8_V1, v2=V8_V2):
    """Never modify the verified original results; validate their recorded provenance."""
    p1 = load_json(v1/'PROTOCOL.json')
    p2 = load_json(v2/'PROTOCOL.json')
    if p1 != p2 or json_hash(p1) != V8_PROTO_SHA:
        fail('Original V8 V1/V2 PROTOCOL.json mismatch or unexpected scientific hash')
    audit = load_json(original_export/'FINAL_DERIVED_AUDIT.json')
    if audit.get('status') != 'DERIVED_770_CHECKSUM_VERIFIED_V1_V2_NOT_OFFICIAL_FROZEN_SUMMARIZE':
        fail('Original V8 derived export does not have its verified completion status')
    expected = {
        'saved_checkpoint_count':770, 'unique_checkpoint_count':770,
        'n_source_seeds':10, 'protocol_sha256':V8_PROTO_SHA,
    }
    for key,value in expected.items():
        if audit.get(key) != value:
            fail(f'Original V8 derived export provenance differs at {key}: {audit.get(key)!r}')
    if audit.get('source_code_fingerprint_match_between_v1_v2_protocols') is not True:
        fail('Original V8 archive did not confirm matching source-code fingerprints')
    digests = audit.get('derived_export_file_sha256', {})
    for filename in ('PARTICIPANT_SEED_AVERAGED.csv', 'METHOD_SUMMARY.csv',
                     'CHECKPOINT_MANIFEST_770.csv'):
        p = original_export/filename
        actual = sha_file(p)
        if digests.get(filename) != actual:
            fail(f'V8 derived export checksum mismatch: {filename}')
    manifest = read_csv(original_export/'CHECKPOINT_MANIFEST_770.csv')
    found = {(r['dataset'], int(r['seed']), r['participant']) for r in manifest}
    want = {(ds, seed, pid) for ds in DATASETS for seed in range(1,11) for pid in PIDS[ds]}
    if len(manifest) != 770 or found != want:
        fail('Original verified manifest is not exactly the required 770 unique evaluations')
    if any((r['source_run'] == 'V1') != (int(r['seed']) <= 5) for r in manifest):
        fail('Original V8 V1/V2 source partitions differ from archived protocol')
    return {
        'verified_770_source_export': str(original_export),
        'source_sha256': {fn:digests[fn] for fn in (
            'PARTICIPANT_SEED_AVERAGED.csv', 'METHOD_SUMMARY.csv', 'CHECKPOINT_MANIFEST_770.csv')},
        'original_protocol_sha256': V8_PROTO_SHA,
        'original_completion_status':audit['status'],
    }


def verify_other_provenance(folder, codehash, name):
    protocol = load_json(folder/'PROTOCOL.json')
    summary = load_json(folder/'FULL_RUN_SUMMARY.json')
    progress = load_json(folder/'PROGRESS_SUMMARY.json')
    for label, item in (('full',summary), ('progress',progress)):
        if item.get('complete') is not True or item.get('n_verified_cases') != 770 or item.get('expected_cases')!=770:
            fail(f'{name} {label}: not 770/770 completed and verified')
    if protocol.get('v8_protocol_sha256')!=V8_PROTO_SHA or protocol.get('code_sha256') != codehash:
        fail(f'{name}: saved scientific protocol differs from frozen code/provenance')
    if progress != summary:
        fail(f'{name}: the latest progress summary differs from full-run summary; check file freshness')
    path = folder/'PARTICIPANT_SEED_AVERAGES.csv'
    return {'full_status': '770/770 reported and verified by its original summarizer',
            'protocol_sha256': json_hash(protocol), 'scientific_code_sha256': codehash,
            'participant_file_sha256': sha_file(path)}


def import_v8_subjects(rows):
    """V8 export is long form: one row per participant×method×mode×metric."""
    chosen = set(METHODS) & {m for m in METHODS if SOURCE[m] == 'V8'}
    req = {(ds,pid,method,mode,metric) for ds in DATASETS for pid in PIDS[ds]
           for method in chosen for mode, metrics in (
               ('shared_gate_timing',METRICS), ('independent_detection',(BA,)))
           for metric in metrics}
    matched = {}
    for r in rows:
        method = r.get('method')
        if method not in chosen:
            continue
        mode = r.get('mode')
        metric = r.get('metric')
        if (mode=='shared_gate_timing' and metric in METRICS) or (mode=='independent_detection' and metric==BA):
            key = (r.get('dataset'),r.get('participant'),method,mode,metric)
            if key not in req:
                fail(f'Unexpected V8 participant/method/metric: {key}')
            if key in matched:
                fail(f'Duplicate V8 participant metric {key}')
            if int(r['n_source_seeds']) != SEEDS or int(r['n_finite_source_seeds']) != SEEDS:
                fail(f'V8 {key}: fewer than 10 finite source-seed evaluations')
            matched[key] = parse_finite(r['mean_across_source_seeds'], f'V8 {key}')
    check_exact_subjects(matched,req,'V8 published comparison inputs')
    return matched


def import_paper_subjects(rows,name,expected_methods):
    expected = {(ds,pid,method,mode,metric) for ds in DATASETS for pid in PIDS[ds]
                for method in expected_methods for mode, metrics in (
                    ('shared_gate_timing',METRICS), ('independent_detection',(BA,)))
                for metric in metrics}
    matched = {}
    # Summaries additionally contain methods from original V8; use those for parity tests.
    for r in rows:
        ds,pid,method,mode = (r.get('dataset'),r.get('participant'),r.get('method'),r.get('mode'))
        if ds not in DATASETS or pid not in PIDS[ds]:
            fail(f'{name} unexpected participant: {(ds,pid)}')
        if int(r['n_seeds']) != SEEDS:
            fail(f'{name} incomplete 10-seed participant/method/mode: {(ds,pid,method,mode)}')
        for metric in ((METRICS if mode=='shared_gate_timing' else (BA,) if mode=='independent_detection' else ())):
            v = r.get(metric)
            if method in expected_methods or method in (HB,'EXTRATREES9_CONTROL','RIDGE9_CONTROL'):
                key = (ds,pid,method,mode,metric)
                if key in matched:
                    fail(f'Duplicate {name} participant metric: {key}')
                matched[key] = parse_finite(v,f'{name} {key}')
    check_exact_subjects({k:v for k,v in matched.items() if k in expected}, expected, name)
    return matched


def validate_against_saved_full_summary(rows, participant_csv, name):
    """Prevent old/mixed summary files that do not match the uploaded paper table."""
    values = defaultdict(list)
    for (ds, pid, method, mode, metric), val in participant_csv.items():
        values[ds,method,mode,metric].append(val)
    for r in rows:
        ds,method,mode = r.get('dataset'),r.get('method'),r.get('mode')
        if ds not in DATASETS:
            continue
        for metric in (*METRICS, BA):
            key = (ds,method,mode,metric)
            if key not in values:
                continue
            v = r.get(metric)
            if v is None:
                fail(f'{name} full summary lacks expected metric {key}')
            source_mean = float(np.mean(values[key]))
            # Full output is participant-averaged with all 10 source seeds.
            if not math.isclose(float(v), source_mean, abs_tol=1e-8, rel_tol=1e-8):
                fail(f'{name} summary/participant mismatch {key}: {v} vs {source_mean}')


def validate_baseline_parity(v8, cohen, lina, tol=1e-8):
    """Same participant/endpoint must produce identical saved HemoBoundary controls."""
    checked = Counter()
    for name, obj, baseline_methods in (
        ('Cohen',cohen,(HB,'EXTRATREES9_CONTROL','RIDGE9_CONTROL')),
        ('Lina',lina,(HB,'EXTRATREES9_CONTROL')),
    ):
        for ds in DATASETS:
            for pid in PIDS[ds]:
                for method in baseline_methods:
                    for mode, metrics in (('shared_gate_timing',METRICS),('independent_detection',(BA,))):
                        for metric in metrics:
                            key=(ds,pid,method,mode,metric)
                            if key not in obj or key not in v8:
                                fail(f'Missing cross-study HemoBoundary/ExtraTrees baseline parity check {name} {key}')
                            a,b=v8[key],obj[key]
                            if not math.isclose(a,b,abs_tol=tol,rel_tol=tol):
                                fail(f'Baseline parity failure {name} {key}: verified V8={a}, other={b}')
                            checked[name]+=1
    return dict(checked)


def compile_subject_data(v8, cohen, lina):
    subjects = {}
    for method in METHODS:
        src = {'V8':v8, 'Cohen':cohen, 'Lina':lina}[SOURCE[method]]
        for ds in DATASETS:
            for pid in PIDS[ds]:
                for mode, metrics in (('shared_gate_timing',METRICS),('independent_detection',(BA,))):
                    for metric in metrics:
                        key=(ds,pid,method,mode,metric)
                        if key not in src:
                            fail(f'Incomplete Table 4 method input {key}')
                        subjects[key] = src[key]
    expected_rows = sum(len(PIDS[ds]) for ds in DATASETS) * len(METHODS) * 5
    if len(subjects)!=expected_rows:
        fail(f'Expected {expected_rows} complete participant/metric observations, got {len(subjects)}')
    return subjects


def paired_signflip_and_bootstrap(tests, reps=SIGN_FLIPS, boots=BOOTSTRAPS, seed=RNG_SEED):
    """Use same random sign vectors within each dataset for all 40 contrasts."""
    by_ds = defaultdict(list)
    for t in tests:
        by_ds[t['dataset']].append(t)
    for i, ds in enumerate(DATASETS):
        group = sorted(by_ds[ds], key=lambda t:(t['family'],t['method'],t['metric']))
        n = len(PIDS[ds])
        if len(group)!=len(METHODS[1:])*5:
            fail(f'{ds}: wrong comparison family/endpoint count')
        D=np.stack([t['_diff'] for t in group], axis=1)  # [n,40]
        if D.shape!=(n,40) or not np.isfinite(D).all():
            fail('Paired statistical matrix has wrong shape or nonfinite values')
        rng=np.random.default_rng(seed + i)
        observed=np.abs(D.sum(axis=0))
        exceed=np.zeros(len(group), dtype=np.int64)
        remaining=reps
        while remaining:
            k=min(5000,remaining)
            signs=rng.integers(0,2,size=(k,n),dtype=np.int8).astype(np.float64)*2-1
            val=np.abs(signs @ D)
            exceed+=np.count_nonzero(val >= (observed[None,:]-1e-10),axis=0)
            remaining-=k
        raw_p=(exceed+1.)/(reps+1.)
        # Resample WHOLE PARTICIPANTS, preserving their 10-seed averages and pairing.
        remaining=boots
        means=[]
        while remaining:
            k=min(4000,remaining)
            weights=rng.multinomial(n,[1.0/n]*n,size=k)
            means.append((weights @ D)/n)
            remaining-=k
        sims=np.concatenate(means,axis=0)
        ci=np.percentile(sims,[2.5,97.5],axis=0)
        for j,t in enumerate(group):
            t['raw_p']=float(raw_p[j])
            t['ci95_low']=float(ci[0,j])
            t['ci95_high']=float(ci[1,j])
            t['n_participants']=n

def holm_pvalues(tests):
    for family,size in ((FAMILY_TIMING,96),(FAMILY_BA,24)):
        group=sorted((t for t in tests if t['family']==family), key=lambda x:(x['raw_p'],x['dataset'],x['method'],x['metric']))
        if len(group)!=size:
            fail(f'Unexpected Holm family size {family}: {len(group)} versus {size}')
        high=0.0
        for i,t in enumerate(group):
            high=max(high,(size-i)*float(t['raw_p']))
            t['holm_p']=float(min(1.,high))
            t['holm_family_size']=size
            t['holm_significant_at_0_05']=t['holm_p']<0.05


def paired_tests(subjects,reps=SIGN_FLIPS,boots=BOOTSTRAPS,seed=RNG_SEED):
    tests=[]
    for ds in DATASETS:
        for method in METHODS[1:]:
            for mode,metrics in (('shared_gate_timing', METRICS),('independent_detection',(BA,))):
                family=FAMILY_TIMING if mode=='shared_gate_timing' else FAMILY_BA
                for metric in metrics:
                    hb=np.array([subjects[(ds,p,HB,mode,metric)] for p in PIDS[ds]],float)
                    alt=np.array([subjects[(ds,p,method,mode,metric)] for p in PIDS[ds]],float)
                    diff=(alt-hb) if mode=='shared_gate_timing' else (hb-alt)
                    tests.append({
                        'family': family,'dataset':ds,'method':method,
                        'display_method':DISPLAY[method], 'source':SOURCE[method],
                        'mode':mode,'metric':metric,
                        'boundary':metric.partition('_')[0] if mode=='shared_gate_timing' else 'detection',
                        'n_source_seeds_per_participant':10,
                        'mean_hemoboundary':float(np.mean(hb)),
                        'mean_comparator':float(np.mean(alt)),
                        'mean_gain':float(np.mean(diff)),
                        'gain_definition':'comparator_minus_HB_loss_sec' if mode=='shared_gate_timing' else 'HB_minus_comparator_balanced_accuracy',
                        '_diff':diff,
                    })
    if len(tests)!=120:
        fail(f'Expected 120 paired tests, got {len(tests)}')
    paired_signflip_and_bootstrap(tests,reps,boots,seed)
    holm_pvalues(tests)
    tests.sort(key=lambda t:(t['dataset'],METHODS.index(t['method']),0 if t['mode']=='shared_gate_timing' else 1,t['metric']))
    for t in tests:
        t.pop('_diff')
        t['p_test']='two-sided paired participant sign flip, Monte Carlo plus-one correction'
        t['ci_method']='95% percentile participant bootstrap of the mean gain (not multiplicity-adjusted)'
    return tests


def create_paper_table(subjects, tests):
    by_test={(t['dataset'],t['method'],t['metric']):t for t in tests}
    table=[]
    for ds in DATASETS:
        for method in METHODS:
            d={'dataset':ds,'method':method,'display_method':DISPLAY[method],
               'implementation':'original HemoBoundary' if method==HB else 'classical ML' if SOURCE[method]=='V8' and method in METHODS[1:3] else 'paper-guided short-epoch adaptation',
               'n_participants':len(PIDS[ds]),'n_seeds':10,
               'balanced_accuracy':float(np.mean([subjects[(ds,p,method,'independent_detection',BA)] for p in PIDS[ds]]))}
            for b in TIMING:
                met=f'{b}_loss_missing4_sec'
                d[met]=float(np.mean([subjects[(ds,p,method,'shared_gate_timing',met)] for p in PIDS[ds]]))
            if method!=HB:
                for met in (*METRICS,BA):
                    t=by_test[(ds,method,met)]
                    suffix='ba' if met==BA else met.split('_')[0]
                    d[f'{suffix}_mean_gain']=t['mean_gain']
                    d[f'{suffix}_ci95_low']=t['ci95_low']
                    d[f'{suffix}_ci95_high']=t['ci95_high']
                    d[f'{suffix}_holm_p']=t['holm_p']
            table.append(d)
    return table


def print_short_results(tests):
    print('\n'+'='*117)
    print('HEMOBOUNDARY TABLE 4: PARTICIPANT-PAIRED GAINS (positive favors HemoBoundary)')
    print('Timing = comparator loss − HemoBoundary loss (seconds); BA = HemoBoundary BA − comparator BA')
    print('Two-sided sign flip, 100k draws; 20k subject bootstraps; Holm timing m=96, BA m=24')
    print('='*117)
    for ds in DATASETS:
        print(f'\n{ds} | n={len(PIDS[ds])} independent participants, each averaged across 10 source seeds')
        print(f"{'Comparator':30s} {'Metric':10s} {'Gain':>9s} {'95% CI':>24s} {'Holm p':>11s}")
        print('-'*100)
        for method in METHODS[1:]:
            for label, metric in ([('BA',BA)] + [(b,f'{b}_loss_missing4_sec') for b in TIMING]):
                t=next(x for x in tests if x['dataset']==ds and x['method']==method and x['metric']==metric)
                print(f"{DISPLAY[method][:30]:30s} {label:10s} {t['mean_gain']:+9.4f} "
                      f"[{t['ci95_low']:+8.4f}, {t['ci95_high']:+8.4f}] {t['holm_p']:11.6g}")


def main(root=ROOT, dest_parent=OUTPUT_PARENT, repetitions=SIGN_FLIPS, boots=BOOTSTRAPS,
         seed=RNG_SEED, write=True):
    if repetitions<1000 or boots<1000:
        fail('Use at least 1,000 sign flips and 1,000 bootstrap resamples')
    print('='*107)
    print('HEMOBOUNDARY BSPC TABLE 4 — 120 READ-ONLY PARTICIPANT-PAIRED CONTRASTS')
    print('Original models and checkpoints are NEVER run, trained, or modified.')
    print('='*107,flush=True)
    if not root.is_dir():
        try:
            from google.colab import drive
            drive.mount('/content/gdrive')
        except ImportError as e:
            raise RuntimeError('Google Drive is not mounted; this code expects a Colab runtime') from e
    if not root.is_dir():
        fail(f'Exact project root missing: {root}')
    # The normal call uses only exact paths; fixture overrides are handled in self-tests.
    if root==ROOT:
        export=V8_EXPORTED; v1=V8_V1; v2=V8_V2; cohen_dir=COHEN; lina_dir=LINA
    else:
        export=root/V8_EXPORTED.name
        v1=root/V8_V1.name
        v2=root/V8_V2.name
        cohen_dir=root/COHEN.name
        lina_dir=root/LINA.name
    print('[1/5] Checking original V8 source provenance, export file hashes, 770-case manifest',flush=True)
    prov={'V8':verify_v8_provenance(root,export,v1,v2)}
    print('[2/5] Checking both paper-guided 770-case scientific protocol/summary markers',flush=True)
    prov['Cohen']=verify_other_provenance(cohen_dir,COHEN_CODE_SHA,'Cohen-Adad')
    prov['Lina']=verify_other_provenance(lina_dir,LINA_CODE_SHA,'Lina')
    print('[3/5] Loading ten-seed participant averages, validating 3-dataset method coverage',flush=True)
    v8=import_v8_subjects(read_csv(export/'PARTICIPANT_SEED_AVERAGED.csv'))
    cohen=import_paper_subjects(read_csv(cohen_dir/'PARTICIPANT_SEED_AVERAGES.csv'), 'Cohen',
                    ('COHENADAD2007_SHIFT_GLM_V8_ADAPTER','COHENADAD2007_DERIVATIVE_GLM_V8_ADAPTER'))
    lina=import_paper_subjects(read_csv(lina_dir/'PARTICIPANT_SEED_AVERAGES.csv'), 'Lina',
                    ('LINA2010_WAVELET_PCA1_V8_ADAPTER','LINA2010_WAVELET_NOPCA_V8_ADAPTER'))
    for dir_, source_, name in ((cohen_dir,cohen,'Cohen'),(lina_dir,lina,'Lina')):
        validate_against_saved_full_summary(load_json(dir_/'FULL_RUN_SUMMARY.json')['results'],source_,name)
    prov['baseline_parity_verified_count']=validate_baseline_parity(v8,cohen,lina)
    subjects=compile_subject_data(v8,cohen,lina)
    print('PASS: source methods complete, original baseline values agree across all three exports.',flush=True)
    print('[4/5] Running participant-paired sign flips, bootstrap CIs, and Holm correction',flush=True)
    tests=paired_tests(subjects,repetitions,boots,seed)
    table=create_paper_table(subjects,tests)
    print_short_results(tests)
    if not write:
        return tests,table,prov
    print('\n[5/5] Writing NEW inference-only folder; original sources remain unchanged',flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest=dest_parent/f'BSPC_TABLE4_PAIRED_770_{stamp}'
    dest.mkdir(parents=True,exist_ok=False)
    stat_cols=[k for k in tests[0] if k!='_diff']
    write_csv(dest/'PAIRED_ALL_120_HOLM.csv',tests,stat_cols)
    write_csv(dest/'PAIRED_TIMING_96_HOLM.csv',[t for t in tests if t['family']==FAMILY_TIMING],stat_cols)
    write_csv(dest/'PAIRED_DETECTION_24_HOLM.csv',[t for t in tests if t['family']==FAMILY_BA],stat_cols)
    table_cols=list(dict.fromkeys(k for r in table for k in r))
    write_csv(dest/'TABLE4_DESCRIPTIVE_WITH_PAIRED_GAINS.csv',table,table_cols)
    unit_cols=('dataset','participant','method','mode','metric','mean_across_10_source_seeds')
    subj_rows=[dict(zip(unit_cols,(*key,value))) for key,value in sorted(subjects.items())]
    write_csv(dest/'AUDITED_PARTICIPANT_INPUTS.csv',subj_rows,unit_cols)
    audit={
        'status':'COMPLETE_PAIRED_120_NEW_METHOD_FAMILY',
        'verified_770_v8_cases':True,
        'cohen_full_run_verified_by_its_own_summary':True,
        'lina_full_run_verified_by_its_own_summary':True,
        'methods':[dict(name=m,display_name=DISPLAY[m],source=SOURCE[m]) for m in METHODS],
        'dataset_counts':{ds:len(PIDS[ds]) for ds in DATASETS},
        'n_source_seeds':10,'n_comparisons':len(tests),
        'families':{'timing':{'label':FAMILY_TIMING,'n':96},
                    'detection':{'label':FAMILY_BA,'n':24}},
        'test':'two-sided paired sign flip at participant level; Monte Carlo plus-one',
        'confidence_interval':'two-sided 95% subject bootstrap percentile, unadjusted for multiplicity',
        'holm_adjustment':'separate over 96 timing and 24 detection comparisons; global across all three datasets',
        'mc_draws':repetitions,'bootstraps':boots,'random_seed':seed,
        'input_provenance':prov,'generated_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Paper-guided HRfunc/Lina/Cohen comparisons are short-epoch adaptations. Semi-synthetic known-boundary evaluation; not original published author-software superiority.',
        'original_files_read_only':True,
        'is_new_inference_not_original_V8_formal_summary':True,
    }
    out_files=sorted(p for p in dest.iterdir() if p.is_file())
    audit['output_sha256']={p.name:sha_file(p) for p in out_files}
    (dest/'AUDIT_NEW_PAIRED_STATS.json').write_text(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    print('\nPASS: 120 contrasts; 96 timing Holm and 24 detection Holm; results in:',dest,flush=True)
    print('No models rerun, no original checkpoints or source summaries modified.')
    return tests,table,prov


def selftest():
    """In-memory numerical sanity checks, no Google Drive or user data."""
    d=np.array([1.,1.,1.,1.,1.])
    tests=[]
    for ds in DATASETS:
        for method in METHODS[1:]:
            for mode,metrics in (('shared_gate_timing',METRICS),('independent_detection',(BA,))):
                for metric in metrics:
                    tests.append({'dataset':ds,'method':method,'metric':metric,
                                  'family': FAMILY_TIMING if mode=='shared_gate_timing' else FAMILY_BA,
                                  '_diff':np.full(len(PIDS[ds]),0.06 if metric==BA else 0.4),
                                  'mean_gain':0.06 if metric==BA else 0.4})
    paired_signflip_and_bootstrap(tests,reps=1000,boots=1000,seed=114)
    holm_pvalues(tests)
    assert len(tests)==120
    assert all(0<t['raw_p']<=1 and 0<t['holm_p']<=1 for t in tests)
    assert all(t['ci95_low']<=t['mean_gain']+1e-8<=t['ci95_high']+1e-8 for t in tests)
    assert all(t['holm_family_size']==(96 if t['family']==FAMILY_TIMING else 24) for t in tests)
    # All zero differences must have p=1 under sign flips.
    zero_tests=[]
    for ds in DATASETS:
        for method in METHODS[1:]:
            for metric in (*METRICS,BA):
                zero_tests.append({'dataset':ds,'method':method,'metric':metric,
                                   'family':FAMILY_BA if metric==BA else FAMILY_TIMING,
                                   '_diff':np.zeros(len(PIDS[ds])), 'mean_gain':0.0})
    paired_signflip_and_bootstrap(zero_tests,reps=1000,boots=1000,seed=114)
    assert all(t['raw_p']==1.0 for t in zero_tests)
    print('PASS: 120 contrasts, two Holm families, paired test/CI sign conventions, exact-zero null.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='HemoBoundary Table 4 read-only paired statistical analysis')
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--sign-flips',type=int,default=SIGN_FLIPS)
    parser.add_argument('--bootstraps',type=int,default=BOOTSTRAPS)
    args=parser.parse_args()
    if args.selftest:selftest()
    else:main(repetitions=args.sign_flips,boots=args.bootstraps)
