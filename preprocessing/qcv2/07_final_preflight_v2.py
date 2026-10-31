import json
from common_qcv2 import *

def main():
    audit=json.loads((RESULTS/"02_full_audit_summary.json").read_text())
    labels=json.loads((RESULTS/"03_label_integrity_summary.json").read_text())
    stages=json.loads((RESULTS/"04_stage_manifests_summary.json").read_text())
    decode=json.loads((RESULTS/"01_decode_repair_summary.json").read_text())
    protocol=json.loads((RESULTS/"06_protocol_roster_audit.json").read_text())

    global_blockers=[]
    if audit["structurally_bad"]>0:
        global_blockers.append(f'{audit["structurally_bad"]} structurally bad H5')
    if audit["missing_h5"]>0:
        global_blockers.append(f'{audit["missing_h5"]} recordings still missing H5 after decode repair')
    if labels["bad_groups"]>0:
        global_blockers.append(f'{labels["bad_groups"]} inconsistent subject-state label groups')
    if not protocol["original_roster_ok"]:
        global_blockers.append("Original 600-subject roster failed")

    stage1_blockers=list(global_blockers)
    if stages["stage1"]["missing_subjects"]>0:
        stage1_blockers.append(f'{stages["stage1"]["missing_subjects"]} subjects have no Stage-1 READY recording')

    # Stage 2 can scientifically train on available GOOD-sync recordings while
    # retaining the original frozen subject roster. Missing subjects/states are
    # explicitly reported and must not be silently counted in waveform metrics.
    stage2_warnings=[]
    if stages["stage2"]["missing_subjects"]>0:
        stage2_warnings.append(
            f'{stages["stage2"]["missing_subjects"]} subjects have no GOOD-sync Stage-2 recording; '
            'retain them in frozen protocol but they contribute no supervised Stage-2 samples'
        )
    if stages["stage2"]["missing_subject_states"]>0:
        stage2_warnings.append(
            f'{stages["stage2"]["missing_subject_states"]} subject-state groups missing from Stage-2 supervision'
        )

    stage3_blockers=list(global_blockers)
    if stages["stage3"]["missing_subjects"]>0:
        stage3_blockers.append(f'{stages["stage3"]["missing_subjects"]} subjects have no Stage-3 READY recording')

    result={
        "decode_repair":decode,
        "stage1":{
            "training_authorized":len(stage1_blockers)==0,
            "blockers":stage1_blockers,
            "manifest":str(RESULTS/"04_stage1_manifest.csv"),
            **stages["stage1"],
        },
        "stage2":{
            "training_authorized":len(global_blockers)==0 and stages["stage2"]["ready_recordings"]>0,
            "blockers":global_blockers,
            "warnings":stage2_warnings,
            "manifest":str(RESULTS/"04_stage2_manifest.csv"),
            **stages["stage2"],
        },
        "stage3":{
            "training_authorized":len(stage3_blockers)==0,
            "blockers":stage3_blockers,
            "manifest":str(RESULTS/"04_stage3_manifest.csv"),
            **stages["stage3"],
        },
        "critical_policy":[
            "Never regenerate the subject split from QC-filtered subjects.",
            "Use the original 600-subject frozen protocol.",
            "Stage 1 ignores synchronization quality as a gate.",
            "Stage 2 uses only GOOD authoritative ppg_sync recordings for supervision.",
            "Stage 3 does not use source ppg_sync quality as an inclusion gate.",
            "HOLD recordings are never silently treated as READY for the corresponding stage."
        ]
    }

    save_json(RESULTS/"07_final_preflight_v2.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
