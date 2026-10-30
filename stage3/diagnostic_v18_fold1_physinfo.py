#!/usr/bin/env python3
"""
v18 Fold-1 fixed PhysMorph information diagnostic (DEVELOPMENT ONLY)

Tests whether the 12 NEW fixed morphology descriptors add individualized ΔSBP
information beyond the first 4 PRE-Z scale descriptors.

IMPORTANT: v18 PhysMorph16 already contains the legacy PRE-Z4 as dimensions 0..3.
Therefore the scientifically correct ablation is:
  A) SCALE4      = descriptors 0..3
  B) MORPH12     = descriptors 4..15 (new non-scale morphology only)
  C) ALL16       = descriptors 0..15

Each state/pair is summarized exactly in the v18 style:
  mean(B), SD(B), mean(A), SD(A), mean(A-B), SD(A-B)
with exact (camera, clip_index) matching.

Diagnostic RidgeCV is fit on Fold-1 TRAIN only and evaluated on Fold-1 VAL only.
No Stage3 retraining. Blind is never loaded/evaluated.
"""
from pathlib import Path
import json, random
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from config_stage3 import *
from protocol_stage3 import load_frozen_protocol
from data_stage3 import (
    load_stage3_manifest, read_labels_for_allowed_refs,
    TargetScaler, AmpFeatureScaler, PhysMorphFeatureScaler,
    Stage3SubjectStateDataset,
)

OUT = Path('diagnostic_v18_fold1_physinfo')
OUT.mkdir(exist_ok=True)
ALPHAS = np.logspace(-4, 6, 31)


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def role(s):
    s=str(s).lower()
    if 'before' in s: return 0
    if 'after' in s: return 1
    return None


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    mae=float(np.mean(np.abs(p-y)))
    r=float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>1e-12 and np.std(p)>1e-12 else float('nan')
    vy=float(np.var(y)); vp=float(np.var(p))
    ccc=float(2*np.cov(y,p,ddof=0)[0,1]/(vy+vp+(y.mean()-p.mean())**2+1e-12))
    return {'n':int(len(y)),'mae':mae,'r':r,'ccc':ccc,
            'stdR':float(np.std(p)/(np.std(y)+1e-12)),
            'bias':float(np.mean(p-y))}


def state_map(ds):
    m={}
    for i,(key,_) in enumerate(ds.items):
        sid,st=key; m.setdefault(str(sid),{})[str(st)]=i
    return m


def pair_summary(B,A):
    D=A-B
    return np.concatenate([B.mean(0),B.std(0),A.mean(0),A.std(0),D.mean(0),D.std(0)])


def build_rows(ds):
    rows=[]
    for sid,states in sorted(state_map(ds).items()):
        bi=ai=None
        for st,i in states.items():
            rr=role(st)
            if rr==0: bi=i
            elif rr==1: ai=i
        if bi is None or ai is None: continue
        b=ds[bi]; a=ds[ai]
        kb={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(b['cameras'],b['clip_indices']))}
        ka={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(a['cameras'],a['clip_indices']))}
        keys=sorted(set(kb)&set(ka))
        if len(keys)<V14_MIN_MATCHED_SEGMENTS: continue
        ib=[kb[k] for k in keys]; ia=[ka[k] for k in keys]
        B=b['phys_features'][ib].numpy().astype(np.float64)
        A=a['phys_features'][ia].numpy().astype(np.float64)
        if B.shape[1]!=16 or A.shape[1]!=16: raise RuntimeError(f'Expected 16D PhysMorph, got {B.shape}, {A.shape}')
        rows.append({
            'sid':sid,
            'y':float(a['target_raw'][0])-float(b['target_raw'][0]),
            'scale4':pair_summary(B[:,:4],A[:,:4]),
            'morph12':pair_summary(B[:,4:],A[:,4:]),
            'all16':pair_summary(B,A),
            'nmatch':len(keys),
        })
    return rows


def fit_eval(tr,va,key):
    Xtr=np.stack([r[key] for r in tr]); ytr=np.asarray([r['y'] for r in tr])
    Xva=np.stack([r[key] for r in va]); yva=np.asarray([r['y'] for r in va])
    # A second TRAIN-only standardization is appropriate for Ridge because the
    # six summary statistics have different scales even though segment features
    # were already standardized using TRAIN only.
    ss=StandardScaler().fit(Xtr)
    reg=RidgeCV(alphas=ALPHAS,fit_intercept=True).fit(ss.transform(Xtr),ytr)
    pred=reg.predict(ss.transform(Xva))
    return metrics(yva,pred),float(reg.alpha_),pred


def descriptor_screen(trds,vads):
    # Rebuild each individual descriptor's 6-stat representation directly.
    tr=build_rows_full(trds); va=build_rows_full(vads)
    out=[]
    for j in range(16):
        key=f'd{j:02d}'
        m,a,_=fit_eval(tr,va,key)
        out.append({'descriptor_index':j,'alpha':a,**m})
    return out


def build_rows_full(ds):
    rows=[]
    for sid,states in sorted(state_map(ds).items()):
        bi=ai=None
        for st,i in states.items():
            rr=role(st)
            if rr==0: bi=i
            elif rr==1: ai=i
        if bi is None or ai is None: continue
        b=ds[bi]; a=ds[ai]
        kb={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(b['cameras'],b['clip_indices']))}
        ka={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(a['cameras'],a['clip_indices']))}
        keys=sorted(set(kb)&set(ka))
        if len(keys)<V14_MIN_MATCHED_SEGMENTS: continue
        B=b['phys_features'][[kb[k] for k in keys]].numpy().astype(np.float64)
        A=a['phys_features'][[ka[k] for k in keys]].numpy().astype(np.float64)
        row={'sid':sid,'y':float(a['target_raw'][0])-float(b['target_raw'][0]),'nmatch':len(keys)}
        for j in range(16): row[f'd{j:02d}']=pair_summary(B[:,j:j+1],A[:,j:j+1])
        rows.append(row)
    return rows


def main():
    seed_all()
    print('v18 FOLD-1 PHYSMORPH INFORMATION-CONTENT DIAGNOSTIC')
    print('BLIND = LOCKED / NOT OPENED / NOT EVALUATED')
    print('No Stage3 retraining. RidgeCV: TRAIN fit -> VAL evaluation only.')
    print('Ablation: SCALE4 vs NEW MORPH12 vs ALL16.')

    split,folds,_=load_frozen_protocol()
    f1=[f for f in folds['folds'] if int(f['fold'])==1][0]
    refs,_,_=load_stage3_manifest()
    allowed=set(map(str,split['development_subjects']))
    labels=read_labels_for_allowed_refs(refs,allowed)
    ts=TargetScaler().fit(labels,f1['train_subjects'])
    amp=AmpFeatureScaler().fit(CACHE_ROOT/'fold_01',refs,f1['train_subjects'])
    phys=PhysMorphFeatureScaler().fit(CACHE_ROOT/'fold_01',refs,f1['train_subjects'])

    trds=Stage3SubjectStateDataset(CACHE_ROOT/'fold_01',refs,labels,ts,f1['train_subjects'],amp,phys)
    vads=Stage3SubjectStateDataset(CACHE_ROOT/'fold_01',refs,labels,ts,f1['val_subjects'],amp,phys)
    tr=build_rows(trds); va=build_rows(vads)
    print(f'matched subjects: train={len(tr)} val={len(va)}')
    print(f'mean exact matched segments/VAL subject={np.mean([r["nmatch"] for r in va]):.3f}')

    result={'blind_used':False,'fold':1,'train_pairs':len(tr),'val_pairs':len(va),'models':{}}
    specs=[('scale4','SCALE4 legacy PRE-Z (24D)'),('morph12','NEW MORPH12 only (72D)'),('all16','ALL16 SCALE4+MORPH12 (96D)')]
    print('\n=== GROUP ABLATION ===')
    for key,label in specs:
        m,a,_=fit_eval(tr,va,key)
        result['models'][key]={'label':label,'metrics':m,'alpha':a}
        print(f'{label:32s} | MAE={m["mae"]:.4f} r={m["r"]:+.4f} CCC={m["ccc"]:+.4f} stdR={m["stdR"]:.4f} alpha={a:g}')

    screen=descriptor_screen(trds,vads)
    result['single_descriptor_screen']=screen
    print('\n=== SINGLE DESCRIPTOR SCREEN ===')
    print('Indices 0..3 = legacy scale; 4..15 = new morphology.')
    for x in sorted(screen,key=lambda q:(-1e9 if not np.isfinite(q['r']) else q['r']),reverse=True):
        print(f'D{x["descriptor_index"]:02d} | r={x["r"]:+.4f} MAE={x["mae"]:.4f} CCC={x["ccc"]:+.4f} stdR={x["stdR"]:.4f} alpha={x["alpha"]:g}')

    with open(OUT/'summary.json','w') as f: json.dump(result,f,indent=2,allow_nan=True)
    print(f'\nSaved: {OUT/"summary.json"}')
    print('BLIND = LOCKED / NOT OPENED / NOT EVALUATED')

if __name__=='__main__': main()
