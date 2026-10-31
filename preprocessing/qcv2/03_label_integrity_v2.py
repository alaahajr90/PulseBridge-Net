from collections import defaultdict
import json
from common_qcv2 import *

def main():
    ensure_dirs()
    rows = load_csv(DB_CSV)
    groups = defaultdict(list)
    for r in rows:
        groups[(sid(r["patient_id"]), str(r["step"]).strip())].append(r)

    cols = [
        "upper_ap","lower_ap","rigidity",
        "weight","height","bmi","age","sex",
        "saturation","temperature","hemoglobin","glycated_hemoglobin",
        "cholesterol","respiratory","pulse","stress"
    ]

    out, bad = [], []
    for (subject,state), xs in sorted(groups.items()):
        q={"subject_id":subject,"state":state,"camera_rows":len(xs)}
        ok=(len(xs)==3)
        for c in cols:
            vals=[str(x.get(c,"")).strip() for x in xs]
            uniq=sorted(set(vals))
            q[c+"_unique"]="|".join(uniq)
            q[c+"_consistent"]=len(uniq)==1
            ok=ok and len(uniq)==1
        q["all_consistent"]=ok
        out.append(q)
        if not ok: bad.append(q)

    write_csv(RESULTS/"03_label_integrity_all.csv",out)
    write_csv(RESULTS/"03_label_integrity_bad.csv",bad)
    summary={
        "subject_state_groups":len(out),
        "expected":EXPECTED_SUBJECT_STATES,
        "bad_groups":len(bad),
    }
    save_json(RESULTS/"03_label_integrity_summary.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
