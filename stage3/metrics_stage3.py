from __future__ import annotations
from collections import defaultdict
import numpy as np
from config_stage3 import BOOTSTRAP_N,BOOTSTRAP_ALPHA,SEED

def _r(y,p):return float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>1e-12 and np.std(p)>1e-12 else float("nan")
def regression_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;mae=float(np.mean(np.abs(e)));mse=float(np.mean(e*e));rmse=float(np.sqrt(mse));bias=float(np.mean(e));sd=float(np.std(e,ddof=1)) if len(e)>1 else 0.
    sst=np.sum((y-y.mean())**2);r2=float(1-np.sum(e*e)/sst) if sst>1e-12 else float("nan");r=_r(y,p);vx=np.var(y);vy=np.var(p);ccc=float(2*np.cov(y,p,ddof=0)[0,1]/(vx+vy+(y.mean()-p.mean())**2+1e-12))
    true_std=float(np.std(y,ddof=0));pred_std=float(np.std(p,ddof=0));std_ratio=float(pred_std/(true_std+1e-12));mean_baseline_mae=float(np.mean(np.abs(y-y.mean())))
    return {"mae":mae,"mse":mse,"rmse":rmse,"pearson_r":r,"ccc":ccc,"r2":r2,"bias":bias,"error_sd":sd,"loa_low":bias-1.96*sd,"loa_high":bias+1.96*sd,
            "true_std":true_std,"pred_std":pred_std,"std_ratio":std_ratio,"mean_baseline_mae":mean_baseline_mae,"mae_gain_vs_mean":mean_baseline_mae-mae}

def paired_delta_metrics(rows,target):
    by=defaultdict(dict)
    for r in rows:by[str(r["subject_id"])][str(r["state"]).lower()]=r
    td=[];pd=[]
    for sid,d in by.items():
        if "before" in d and "after" in d:
            td.append(d["after"][f"true_{target}"]-d["before"][f"true_{target}"]);pd.append(d["after"][f"pred_{target}"]-d["before"][f"pred_{target}"])
    if not td:return {"n_pairs":0}
    td=np.asarray(td,float);pd=np.asarray(pd,float)
    return {"n_pairs":len(td),"delta_mae":float(np.mean(np.abs(pd-td))),"delta_r":_r(td,pd),"true_delta_mean":float(td.mean()),"pred_delta_mean":float(pd.mean()),"delta_std_ratio":float(np.std(pd)/(np.std(td)+1e-12))}

def all_target_metrics(rows):
    out={}
    for target in ("sbp","dbp","stiffness"):
        y=np.array([r[f"true_{target}"] for r in rows]);p=np.array([r[f"pred_{target}"] for r in rows])
        for k,v in regression_metrics(y,p).items():out[f"{target}_{k}"]=v
        for k,v in paired_delta_metrics(rows,target).items():out[f"{target}_{k}"]=v
    out["n_subject_states"]=len(rows);out["n_subjects"]=len({r["subject_id"] for r in rows});return out

def subject_cluster_bootstrap(rows,n=BOOTSTRAP_N,alpha=BOOTSTRAP_ALPHA,seed=SEED):
    rng=np.random.default_rng(seed);by=defaultdict(list)
    for r in rows:by[str(r["subject_id"])].append(r)
    sids=np.array(sorted(by));stat_keys=[f"{t}_{m}" for t in ("sbp","dbp","stiffness") for m in ("mae","rmse","pearson_r","ccc","std_ratio")];vals={k:[] for k in stat_keys}
    for _ in range(n):
        sample=rng.choice(sids,size=len(sids),replace=True);boot=[]
        for j,sid in enumerate(sample):
            for r in by[sid]:q=dict(r);q["subject_id"]=f"{sid}__boot{j}";boot.append(q)
        m=all_target_metrics(boot)
        for k in stat_keys:
            if np.isfinite(m[k]):vals[k].append(m[k])
    lo=alpha/2;hi=1-alpha/2;return {k:{"low":float(np.quantile(v,lo)),"high":float(np.quantile(v,hi)),"n":len(v)} for k,v in vals.items() if v}

def rows_from_batch(metas,truth,pred,camera_weight_bp=None,camera_weight_st=None):
    rows=[]
    for i,m in enumerate(metas):
        q={"subject_id":m["subject_id"],"state":m["state"],"true_sbp":float(truth[i,0]),"true_dbp":float(truth[i,1]),"true_stiffness":float(truth[i,2]),"pred_sbp":float(pred[i,0]),"pred_dbp":float(pred[i,1]),"pred_stiffness":float(pred[i,2]),"n_segments":int(m["n_segments"])}
        if camera_weight_bp is not None:
            for c,name in enumerate(("FullHDwebcam","IriunWebcam","USBVideo")):q[f"bp_weight_{name}"]=float(camera_weight_bp[i,c]);q[f"stiffness_weight_{name}"]=float(camera_weight_st[i,c])
        rows.append(q)
    return rows

def aggregate_segment_rows_to_recordings(segment_rows):
    by=defaultdict(list)
    for r in segment_rows:by[(r["subject_id"],r["state"],r["camera"])].append(r)
    out=[]
    for (sid,state,camera),xs in sorted(by.items()):
        q={"subject_id":sid,"state":state,"camera":camera,"n_segments":len(xs)}
        for target in ("sbp","dbp","stiffness"):
            q[f"true_{target}"]=float(xs[0][f"true_{target}"]);q[f"pred_{target}"]=float(np.mean([x[f"pred_{target}"] for x in xs]))
        out.append(q)
    return out
