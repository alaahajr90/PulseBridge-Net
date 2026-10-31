import json
from common_qcv2 import *

REQUIRED_FRAGMENTS = [
    'root / "ppg_sync" / f"{sid}_{camera}_{state}.txt"',
    'root / "ppg" / f"{sid}_{state}.PW"',
    'return output_root / f"subject_{sid}" / camera / state',
    'ppg_all, ppg_dt_all = load_ppg_sync(src["ppg_sync"])',
    'n_usable = min(n_meta, n_sync)',
    'rec = extract_recording_rois(',
    '"ppg_sync_dt"',
    'h5.attrs["source_ppg_sync"]',
]

FORBIDDEN_ASSUMPTION = 'load_ppg_raw('

def main():
    ensure_dirs()
    text = EXTRACTOR_PATH.read_text(encoding="utf-8")
    missing = [x for x in REQUIRED_FRAGMENTS if x not in text]

    result = {
        "extractor_path": str(EXTRACTOR_PATH),
        "extractor_sha256": sha256(EXTRACTOR_PATH),
        "contract_pass": not missing,
        "missing_required_fragments": missing,
        "source_ppg_sync_is_authoritative_input": 'load_ppg_sync(src["ppg_sync"])' in text,
        "raw_ppg_not_used_to_build_h5_target_in_process_recording": FORBIDDEN_ASSUMPTION not in text,
        "output_layout": "subject_<sid>/<camera>/<state>/recording.h5",
    }
    save_json(RESULTS/"00_source_contract.json", result)
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit("Extractor contract mismatch. STOP.")

if __name__ == "__main__":
    main()
