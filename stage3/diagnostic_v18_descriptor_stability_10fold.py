#!/usr/bin/env python3
"""
v18 PhysMorph Descriptor Stability — 10 Development Folds

Purpose:
  Test whether each fixed PhysMorph descriptor D00..D15 generalizes across
  all 10 DEVELOPMENT folds, especially D10/D06/D05/D07.

Protocol:
  - NO Stage3 training.
  - BLIND is never loaded/evaluated.
  - For each fold: fit RidgeCV on that fold's TRAIN subjects only.
  - Evaluate on that fold's VAL subjects only.
  - Exact (camera, clip_index) Before/After matching.
  - Each descriptor uses its 6D pair summary:
      B_mean, B_std, A_mean, A_std, Delta_mean, Delta_std
  - Reports per-fold metrics, mean/median/std r, positive-fold count,
    and pooled OOF metrics.
"""

from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from config_stage3 import *
from protocol_stage3 import load_frozen_protocol
from data_stage3 import (
    load_stage3_manifest, read_labels_for_allowed_refs,
    TargetScaler, AmpFeatureScaler, Stage3SubjectStateDataset
)
import data_stage3 as D

OUT = Path("diagnostic_v18_descriptor_stability_10fold")
OUT.mkdir(exist_ok=True)
ALPHAS = np.logspace(-4, 6, 31)

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def state_role(s):
    s=str(s).lower()
    if "before" in s: return 0
    if "after" in s: return 1
    return None

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    mae=float(np.mean(np.abs(p-y)))
    r=float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>1e-12 and np.std(p)>1e-12 else float("nan")
    vy=float(np.var(y)); vp=float(np.var(p))
    ccc=float(2*np.cov(y,p,ddof=0)[0,1]/(vy+vp+(y.mean()-p.mean())**2+1e-12))
    return dict(n=int(len(y)),mae=mae,r=r,ccc=ccc,
                stdR=float(np.std(p)/(np.std(y)+1e-12)),
                bias=float(np.mean(p-y)))

def find_extractor():
    cand=[]
    for name in dir(D):
        obj=getattr(D,name)
        low=name.lower()
        if callable(obj) and ("morph" in low or "phys" in low) and \
           ("extract" in low or "feature" in low or "descriptor" in low):
            cand.append((name,obj))
    cand.sort(key=lambda x:("phys" not in x[0].lower(),
                            "morph" not in x[0].lower(),
                            "fixed" not in x[0].lower(),x[0]))
    return cand

def call_extractor(fn,w):
    trials=[w.detach().cpu().numpy(), w.detach().cpu(),
            w.detach().cpu().numpy().reshape(-1), w.detach().cpu().reshape(-1)]
    last=None
    for x in trials:
        try:
            z=fn(x)
            if torch.is_tensor(z): z=z.detach().cpu().numpy()
            z=np.asarray(z,dtype=np.float64).reshape(-1)
            if z.size==16 and np.isfinite(z).all(): return z
        except Exception as e:
            last=e
    raise RuntimeError(f"Cannot obtain finite 16D PhysMorph: {last}")

def state_map(ds):
    m={}
    for i,(key,_) in enumerate(ds.items):
        sid,st=key
        m.setdefault(str(sid),{})[str(st)]=i
    return m

def build_rows(ds,fn):
    sm=state_map(ds); rows=[]
    for sid,states in sorted(sm.items()):
        bi=ai=None
        for st,i in states.items():
            rr=state_role(st)
            if rr==0: bi=i
            elif rr==1: ai=i
        if bi is None or ai is None: continue
        b=ds[bi]; a=ds[ai]
        kb={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(b["cameras"],b["clip_indices"]))}
        ka={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(a["cameras"],a["clip_indices"]))}
        keys=sorted(set(kb)&set(ka))
        if len(keys)<V14_MIN_MATCHED_SEGMENTS: continue

        B=[]; A=[]
        for k in keys:
            B.append(call_extractor(fn,b["prez_waves"][kb[k]]))
            A.append(call_extractor(fn,a["prez_waves"][ka[k]]))
        B=np.stack(B); A=np.stack(A); DD=A-B

        # 96D = 6 statistics x 16 descriptors
        x=np.concatenate([B.mean(0),B.std(0),A.mean(0),A.std(0),DD.mean(0),DD.std(0)])
        rows.append(dict(sid=sid,
                         y=float(a["target_raw"][0])-float(b["target_raw"][0]),
                         x=x,nmatch=len(keys)))
    return rows

def descriptor_matrix(rows,j):
    # layout: Bmean, Bstd, Amean, Astd, Dmean, Dstd; each block 16D
    idx=[j,16+j,32+j,48+j,64+j,80+j]
    return np.stack([r["x"][idx] for r in rows])

def main():
    seed_all()
    print("v18 PHYSMORPH DESCRIPTOR STABILITY — 10 DEVELOPMENT FOLDS")
    print("BLIND = LOCKED / NOT OPENED / NOT EVALUATED")
    print("No Stage3 retraining. Per fold: TRAIN RidgeCV -> VAL.")
    print("Exact (camera, clip_index) matching.\n")

    split,folds,_=load_frozen_protocol()
    refs,hold,_=load_stage3_manifest()
    allowed=set(map(str,split["development_subjects"]))
    labels=read_labels_for_allowed_refs(refs,allowed)

    # Determine exact extractor on first fold/real waveform.
    f0=folds["folds"][0]
    ts0=TargetScaler().fit(labels,f0["train_subjects"])
    as0=AmpFeatureScaler().fit(CACHE_ROOT/"fold_01",refs,f0["train_subjects"])
    ds0=Stage3SubjectStateDataset(CACHE_ROOT/"fold_01",refs,labels,ts0,f0["train_subjects"],as0)
    candidates=find_extractor()
    fn=name=None
    for n,f in candidates:
        try:
            z=call_extractor(f,ds0[0]["prez_waves"][0])
            name,fn=n,f; break
        except Exception:
            pass
    if fn is None:
        raise RuntimeError("No exact finite 16D PhysMorph extractor found in data_stage3.py")
    print("Using exact v18 extractor:",name)

    per={j:[] for j in range(16)}
    oof={j:{"y":[],"p":[],"sid":[]} for j in range(16)}
    fold_counts=[]

    for f in folds["folds"]:
        fold=int(f["fold"])
        cache=CACHE_ROOT/f"fold_{fold:02d}"
        ts=TargetScaler().fit(labels,f["train_subjects"])
        amps=AmpFeatureScaler().fit(cache,refs,f["train_subjects"])
        trds=Stage3SubjectStateDataset(cache,refs,labels,ts,f["train_subjects"],amps)
        vads=Stage3SubjectStateDataset(cache,refs,labels,ts,f["val_subjects"],amps)
        tr=build_rows(trds,fn); va=build_rows(vads,fn)
        ytr=np.array([r["y"] for r in tr]); yva=np.array([r["y"] for r in va])
        fold_counts.append(dict(fold=fold,train_pairs=len(tr),val_pairs=len(va),
                                val_mean_matched=float(np.mean([r["nmatch"] for r in va]))))
        print(f"\n=== FOLD {fold:02d} | train={len(tr)} val={len(va)} ===")
        for j in range(16):
            Xtr=descriptor_matrix(tr,j); Xva=descriptor_matrix(va,j)
            ss=StandardScaler().fit(Xtr)
            reg=RidgeCV(alphas=ALPHAS,fit_intercept=True).fit(ss.transform(Xtr),ytr)
            pred=reg.predict(ss.transform(Xva))
            m=metrics(yva,pred)
            row=dict(fold=fold,descriptor=j,alpha=float(reg.alpha_),**m)
            per[j].append(row)
            oof[j]["y"].extend(yva.tolist()); oof[j]["p"].extend(pred.tolist())
            oof[j]["sid"].extend([r["sid"] for r in va])
            print(f"D{j:02d} r={m['r']:+.4f} MAE={m['mae']:.4f} "
                  f"CCC={m['ccc']:+.4f} stdR={m['stdR']:.4f} a={float(reg.alpha_):g}")

    summary=[]
    print("\n================ 10-FOLD SUMMARY ================")
    print("Desc | mean_r median_r  sd_r  pos/10 | pooled_r pooled_CCC pooled_MAE pooled_stdR")
    print("-"*92)
    for j in range(16):
        rs=np.array([q["r"] for q in per[j]],float)
        pm=metrics(oof[j]["y"],oof[j]["p"])
        row=dict(
            descriptor=j,
            mean_r=float(np.nanmean(rs)),
            median_r=float(np.nanmedian(rs)),
            std_r=float(np.nanstd(rs)),
            positive_folds=int(np.sum(rs>0)),
            pooled=pm,
            per_fold=per[j]
        )
        summary.append(row)
        print(f"D{j:02d}  | {row['mean_r']:+.4f}  {row['median_r']:+.4f}  "
              f"{row['std_r']:.4f}   {row['positive_folds']:2d}/10 | "
              f"{pm['r']:+.4f}    {pm['ccc']:+.4f}      {pm['mae']:.4f}      {pm['stdR']:.4f}")

    ranking=sorted(summary,key=lambda z:(-999 if not np.isfinite(z["pooled"]["r"]) else z["pooled"]["r"]),reverse=True)
    print("\n=== RANKING BY POOLED OOF r ===")
    for k,z in enumerate(ranking,1):
        print(f"{k:02d}. D{z['descriptor']:02d}: pooled r={z['pooled']['r']:+.4f} | "
              f"mean fold r={z['mean_r']:+.4f} | median={z['median_r']:+.4f} | "
              f"positive={z['positive_folds']}/10")

    result={
        "blind_used":False,
        "extractor":name,
        "protocol":"10-fold development-only; per-fold TRAIN RidgeCV -> VAL; exact camera+clip matching",
        "fold_counts":fold_counts,
        "descriptor_summary":summary,
        "ranking_by_pooled_r":[z["descriptor"] for z in ranking]
    }
    with open(OUT/"summary.json","w") as f:
        json.dump(result,f,indent=2,allow_nan=True)

    # Simple CSV for convenient inspection.
    import csv
    with open(OUT/"per_fold.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["fold","descriptor","alpha","n","mae","r","ccc","stdR","bias"])
        w.writeheader()
        for j in range(16):
            for row in per[j]: w.writerow(row)

    print("\nSaved:",OUT/"summary.json")
    print("Saved:",OUT/"per_fold.csv")
    print("BLIND = LOCKED / NOT OPENED / NOT EVALUATED")

if __name__=="__main__":
    main()
