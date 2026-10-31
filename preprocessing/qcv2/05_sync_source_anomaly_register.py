import json
from common_qcv2 import *

def main():
    rows=load_csv(RESULTS/"02_all_recordings_audit.csv")
    anomalies=[]
    for r in rows:
        if r["sync_qc"]!="GOOD":
            anomalies.append({
                "subject_id":r["subject_id"],
                "state":r["state"],
                "camera":r["camera"],
                "h5_path":r["h5_path"],
                "sync_mean_s":r["sync_mean_s"],
                "sync_max_s":r["sync_max_s"],
                "sync_qc":r["sync_qc"],
                "stage1_effect":"NONE if ROI/structure pass",
                "stage2_effect":"EXCLUDED from supervised waveform loss/evaluation",
                "stage3_effect":"NONE if ROI/structure/labels pass; Stage2 model may still infer rPPG",
                "repair_policy":"DO NOT reconstruct/overwrite authoritative ppg_sync from raw .PW without original synchronization generator"
            })
    write_csv(RESULTS/"05_sync_source_anomalies.csv",anomalies)
    summary={
        "anomalies":len(anomalies),
        "review":sum(r["sync_qc"]=="REVIEW" for r in anomalies),
        "bad":sum(r["sync_qc"]=="BAD" for r in anomalies),
        "ppg_sync_modified":False,
        "policy":"quarantine only for Stage 2 supervision"
    }
    save_json(RESULTS/"05_sync_source_anomalies_summary.json",summary)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
