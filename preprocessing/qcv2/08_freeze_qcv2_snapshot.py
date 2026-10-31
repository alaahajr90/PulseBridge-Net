import json
from common_qcv2 import *

def main():
    targets=[
        RESULTS/"00_source_contract.json",
        RESULTS/"01_decode_repair_report.csv",
        RESULTS/"01_decode_repair_summary.json",
        RESULTS/"02_all_recordings_audit.csv",
        RESULTS/"02_full_audit_summary.json",
        RESULTS/"03_label_integrity_summary.json",
        RESULTS/"04_stage1_manifest.csv",
        RESULTS/"04_stage2_manifest.csv",
        RESULTS/"04_stage3_manifest.csv",
        RESULTS/"04_stage_manifests_summary.json",
        RESULTS/"05_sync_source_anomalies.csv",
        RESULTS/"06_protocol_roster_audit.json",
        RESULTS/"07_final_preflight_v2.json",
    ]
    out={}
    for p in targets:
        if p.exists():
            out[p.name]={"path":str(p),"sha256":sha256(p),"bytes":p.stat().st_size}
    save_json(RESULTS/"08_qcv2_snapshot_hashes.json",out)
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
