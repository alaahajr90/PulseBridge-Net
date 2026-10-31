import json
from pathlib import Path
from common_qcv2 import *

def main():
    db=load_csv(DB_CSV)
    roster=sorted(set(sid(r["patient_id"]) for r in db),key=lambda x:int(x) if x.isdigit() else x)

    result={
        "original_subject_roster_count":len(roster),
        "expected":EXPECTED_SUBJECTS,
        "original_roster_ok":len(roster)==EXPECTED_SUBJECTS,
        "frozen_protocol_exists":False,
    }

    split=STAGE1_PROTOCOL/"frozen_subject_split.json"
    folds=STAGE1_PROTOCOL/"development_10folds.json"

    if split.exists() and folds.exists():
        s=json.loads(split.read_text(encoding="utf-8"))
        dev=set(map(str,s["development_subjects"]))
        blind=set(map(str,s["blind_test_subjects"]))
        allp=dev|blind
        result.update({
            "frozen_protocol_exists":True,
            "development_subjects":len(dev),
            "blind_subjects":len(blind),
            "dev_blind_overlap":len(dev&blind),
            "protocol_subjects_total":len(allp),
            "protocol_equals_original_roster":allp==set(roster),
            "split_sha256":sha256(split),
            "folds_sha256":sha256(folds),
        })

    save_json(RESULTS/"06_protocol_roster_audit.json",result)
    print(json.dumps(result,indent=2))

    if not result["original_roster_ok"]:
        raise SystemExit("Original 600-subject roster is not intact.")
    if result["frozen_protocol_exists"] and not result["protocol_equals_original_roster"]:
        raise SystemExit("Frozen protocol does not equal original subject roster.")

if __name__=="__main__":main()
