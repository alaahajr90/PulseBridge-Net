from collections import defaultdict, Counter
import json
from common_qcv2 import *

def bools(x): return str(x).lower() in ("true","1","yes")

def coverage(rows):
    bys=defaultdict(list); byss=defaultdict(list)
    for r in rows:
        if r["status"]=="READY":
            bys[r["subject_id"]].append(r)
            byss[(r["subject_id"],r["state"])].append(r)
    return {
        "ready_recordings":sum(r["status"]=="READY" for r in rows),
        "hold_recordings":sum(r["status"]!="READY" for r in rows),
        "subjects_with_ready":len(bys),
        "missing_subjects":EXPECTED_SUBJECTS-len(bys),
        "subject_states_with_ready":len(byss),
        "missing_subject_states":EXPECTED_SUBJECT_STATES-len(byss),
        "ready_recordings_per_subject":dict(sorted(Counter(len(v) for v in bys.values()).items())),
        "ready_cameras_per_subject_state":dict(sorted(Counter(len(v) for v in byss.values()).items())),
    }

def main():
    ensure_dirs()
    audit=load_csv(RESULTS/"02_all_recordings_audit.csv")

    stage1=[]; stage2=[]; stage3=[]

    for r in audit:
        structural=bools(r["structural_ok"])
        roi_good=r["roi_qc"]=="GOOD"
        sync_good=r["sync_qc"]=="GOOD"
        labels_ok=bools(r["stage3_labels_finite"])

        base={
            "subject_id":r["subject_id"],
            "state":r["state"],
            "camera":r["camera"],
            "view":r["view"],
            "h5_path":r["h5_path"],
            "n_usable":r["n_usable"],
            "n_clips":r["n_clips"],
            "fallback_ratio":r["fallback_ratio"],
            "sync_mean_s":r["sync_mean_s"],
            "sync_max_s":r["sync_max_s"],
            "sync_qc":r["sync_qc"],
        }

        # Stage 1: self-supervised facial representation.
        # PPG synchronization is deliberately NOT a gate.
        reasons=[]
        if not structural: reasons.append("STRUCTURAL")
        if not roi_good: reasons.append("ROI")
        stage1.append({
            **base,
            "status":"READY" if not reasons else "HOLD",
            "hold_reasons":";".join(reasons),
            "sync_used_as_gate":False,
            "role":"facial ROI self-supervised representation"
        })

        # Stage 2: supervised rPPG waveform learning.
        reasons=[]
        if not structural: reasons.append("STRUCTURAL")
        if not roi_good: reasons.append("ROI")
        if not sync_good: reasons.append("SYNC")
        stage2.append({
            **base,
            "status":"READY" if not reasons else "HOLD",
            "hold_reasons":";".join(reasons),
            "sync_used_as_gate":True,
            "role":"ROI -> ground-truth ppg_sync supervised rPPG"
        })

        # Stage 3: predicted rPPG -> BP / rigidity labels.
        # Source ppg_sync quality is NOT a Stage-3 target requirement.
        reasons=[]
        if not structural: reasons.append("STRUCTURAL")
        if not roi_good: reasons.append("ROI")
        if not labels_ok: reasons.append("LABELS")
        stage3.append({
            **base,
            "systolic_bp":r["systolic_bp"],
            "diastolic_bp":r["diastolic_bp"],
            "rigidity":r["rigidity"],
            "status":"READY" if not reasons else "HOLD",
            "hold_reasons":";".join(reasons),
            "sync_used_as_gate":False,
            "role":"predicted rPPG -> SBP/DBP + arterial stiffness"
        })

    write_csv(RESULTS/"04_stage1_manifest.csv",stage1)
    write_csv(RESULTS/"04_stage2_manifest.csv",stage2)
    write_csv(RESULTS/"04_stage3_manifest.csv",stage3)

    summary={
        "stage1":coverage(stage1),
        "stage2":coverage(stage2),
        "stage3":coverage(stage3),
        "scientific_rule":{
            "stage1":"STRUCTURAL + ROI; sync is not a gate",
            "stage2":"STRUCTURAL + ROI + GOOD ppg_sync",
            "stage3":"STRUCTURAL + ROI + finite BP/rigidity labels; source sync is not a gate"
        }
    }
    save_json(RESULTS/"04_stage_manifests_summary.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
