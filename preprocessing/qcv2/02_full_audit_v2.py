import json
from pathlib import Path
from common_qcv2 import *

def main():
    ensure_dirs()
    rows, bad = [], []

    for hp in sorted(CLIP_ROOT.glob("subject_*/*/*/recording.h5")):
        try:
            a = structural_h5_audit(hp)
            a["roi_qc"] = "GOOD" if a["fallback_ratio"] <= ROI_FALLBACK_MAX else "HOLD"
            a["sync_qc"] = sync_class(a["sync_mean_s"], a["sync_max_s"])
            a["stage3_labels_finite"] = bool(
                np.isfinite([a["systolic_bp"], a["diastolic_bp"], a["rigidity"]]).all()
            )
            rows.append(a)
            if not a["structural_ok"]:
                bad.append(a)
        except Exception as e:
            bad.append({"h5_path": str(hp), "error": repr(e)})

    write_csv(RESULTS/"02_all_recordings_audit.csv", rows)
    write_csv(RESULTS/"02_structurally_bad.csv", bad)
    write_csv(RESULTS/"02_roi_hold.csv", [r for r in rows if r["roi_qc"]=="HOLD"])
    write_csv(RESULTS/"02_sync_review.csv", [r for r in rows if r["sync_qc"]=="REVIEW"])
    write_csv(RESULTS/"02_sync_bad.csv", [r for r in rows if r["sync_qc"]=="BAD"])

    summary = {
        "h5_found": len(rows),
        "expected": EXPECTED_RECORDINGS,
        "missing_h5": EXPECTED_RECORDINGS-len(rows),
        "structurally_bad": len(bad),
        "subjects_present": len(set(r["subject_id"] for r in rows)),
        "roi_good": sum(r["roi_qc"]=="GOOD" for r in rows),
        "roi_hold": sum(r["roi_qc"]=="HOLD" for r in rows),
        "sync_good": sum(r["sync_qc"]=="GOOD" for r in rows),
        "sync_review": sum(r["sync_qc"]=="REVIEW" for r in rows),
        "sync_bad": sum(r["sync_qc"]=="BAD" for r in rows),
        "stage3_labels_finite": sum(r["stage3_labels_finite"] for r in rows),
    }
    save_json(RESULTS/"02_full_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
