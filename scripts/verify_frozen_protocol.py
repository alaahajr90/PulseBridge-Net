#!/usr/bin/env python3
"""Verify already-frozen subject protocol WITHOUT regenerating or revealing IDs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

def normalize(v):
    return str(int(float(v))) if str(v).replace('.', '', 1).isdigit() else str(v).strip()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--folds', type=Path, required=True)
    ap.add_argument('--roster-csv', type=Path,
                    help='Optional original MCD db.csv for roster comparison')
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    folds = json.loads(args.folds.read_text())
    dev = set(map(normalize, split['development_subjects']))
    blind = set(map(normalize, split['blind_test_subjects']))
    assert len(dev)==540, 'Development must contain 540 unique subjects'
    assert len(blind)==60, 'Independent test must contain 60 unique subjects'
    assert not dev & blind, 'Development and blind test overlap'
    assert len(dev|blind)==600
    assert int(folds['k_folds'])==10 and len(folds['folds'])==10
    seen=Counter()
    for f in folds['folds']:
        tr=set(map(normalize,f['train_subjects']))
        va=set(map(normalize,f['val_subjects']))
        assert len(tr)==486 and len(va)==54, 'Expected fold 486/54'
        assert not tr&va and not tr&blind and not va&blind
        assert tr|va==dev
        seen.update(va)
    assert set(seen)==dev and set(seen.values())=={1}
    if args.roster_csv:
        with args.roster_csv.open(encoding='utf-8-sig',newline='') as f:
            roster={normalize(x['patient_id']) for x in csv.DictReader(f)}
        assert roster==dev|blind, 'Frozen split differs from original roster'
    print('FROZEN PROTOCOL VALIDATION: PASS')
    print('subjects=600 development=540 independent=60 folds=10')
    print('train_per_fold=486 validation_per_fold=54')
    print('split_sha256='+sha256(args.split))
    print('folds_sha256='+sha256(args.folds))
    print('participant_identifiers_exposed=NO')

if __name__=='__main__':
    main()
