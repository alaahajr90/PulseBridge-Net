from __future__ import annotations
import json, os, shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

from common_qcv2 import *


def count_decodable_frames(video_path):
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            n += 1
    finally:
        cap.release()
    return n


def nonempty_lines(path):
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        return [ln for ln in f if ln.strip()]


def symlink_force(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def patch_provenance_and_counts(h5_path, done_path, src, n_meta_orig, n_sync_orig, n_decodable, n_usable):
    with h5py.File(h5_path, "r+") as h:
        h.attrs["source_video"] = str(src["video"])
        h.attrs["source_meta"] = str(src["meta"])
        h.attrs["source_ppg_sync"] = str(src["ppg_sync"])
        h.attrs["source_ppg_raw"] = str(src["ppg_raw"])
        h.attrs["source_ecg"] = str(src["ecg"])
        h.attrs["source_db_csv"] = str(DB_CSV)

        # Restore counts to describe ORIGINAL source files, while n_usable reflects
        # the valid common prefix including actual sequential video decodability.
        h.attrs["n_meta"] = int(n_meta_orig)
        h.attrs["n_sync"] = int(n_sync_orig)
        h.attrs["n_usable"] = int(n_usable)
        h.attrs["tail_frames"] = int(n_meta_orig - n_usable)

        h.attrs["decode_repair_applied"] = True
        h.attrs["decode_repair_version"] = "Data_QC_v2"
        h.attrs["actual_decodable_frames"] = int(n_decodable)
        h.attrs["decode_repair_rule"] = "min(n_meta,n_sync,n_actual_decodable)"

    info = json.loads(done_path.read_text(encoding="utf-8"))
    info["n_meta"] = int(n_meta_orig)
    info["n_sync"] = int(n_sync_orig)
    info["n_usable"] = int(n_usable)
    info["tail_frames"] = int(n_meta_orig - n_usable)
    info["decode_repair_applied"] = True
    info["decode_repair_version"] = "Data_QC_v2"
    info["actual_decodable_frames"] = int(n_decodable)
    info["decode_repair_rule"] = "min(n_meta,n_sync,n_actual_decodable)"
    mod = load_extractor_module()
    mod.atomic_write_text(done_path, json.dumps(info, indent=2, ensure_ascii=False))


def main():
    ensure_dirs()
    guard_disk()

    mod = load_extractor_module()
    db = mod.DBTable(DB_CSV)
    extractor = mod.ROIExtractor(PREDICTOR_PATH, ROI_SIZE, rgb_store=False)

    errors = load_csv(ERRORS_CSV)
    print(f"Decode-error queue: {len(errors)} recordings")

    report = []

    for i, er in enumerate(errors, 1):
        subject = sid(er["subject_id"])
        state = er["state"]
        camera = er["camera"]

        print("\n" + "-"*108)
        print(f"[{i:02d}/{len(errors):02d}] {subject} | {state} | {camera}")

        row = {
            "subject_id": subject,
            "state": state,
            "camera": camera,
            "original_error": er.get("error",""),
        }

        try:
            guard_disk()
            src = mod.required_paths(DATA_ROOT, subject, state, camera)
            mod.validate_source_paths(src)

            out_dir = mod.recording_output_dir(CLIP_ROOT, subject, camera, state)
            h5_path = out_dir / mod.RECORDING_FILENAME
            done_path = out_dir / mod.DONE_FILENAME

            if h5_path.exists() and done_path.exists():
                row["status"] = "ALREADY_COMPLETE"
                row["h5_path"] = str(h5_path)
                print("  already complete -> skip")
                report.append(row)
                write_csv(RESULTS/"01_decode_repair_report.csv", report)
                continue

            # Never destroy an unexplained existing H5.
            if h5_path.exists() and not done_path.exists():
                row["status"] = "BLOCKED_EXISTING_H5_WITHOUT_DONE"
                row["h5_path"] = str(h5_path)
                print("  BLOCKED: H5 exists without _DONE.json")
                report.append(row)
                write_csv(RESULTS/"01_decode_repair_report.csv", report)
                continue

            meta_lines = nonempty_lines(src["meta"])
            sync_lines = nonempty_lines(src["ppg_sync"])
            n_meta_orig = len(meta_lines)
            n_sync_orig = len(sync_lines)
            n_decodable = count_decodable_frames(src["video"])
            n_usable = min(n_meta_orig, n_sync_orig, n_decodable)

            row.update({
                "n_meta_original": n_meta_orig,
                "n_sync_original": n_sync_orig,
                "n_actual_decodable": n_decodable,
                "n_usable_repaired": n_usable,
                "decode_shortfall_vs_old_rule": min(n_meta_orig,n_sync_orig)-n_usable,
            })

            if n_usable < CLIP_LEN:
                raise RuntimeError(f"Repaired n_usable={n_usable} < clip_len={CLIP_LEN}")

            # Temporary scientific-source mirror:
            # exact original extractor is reused; only meta/ppg_sync are truncated
            # to the proven decodable common prefix. Raw sources remain untouched.
            tmp = TEMP_ROOT / f"{subject}_{camera}_{state}"
            if tmp.exists():
                shutil.rmtree(tmp)
            for d in ("video","meta","ppg_sync","ppg","ecg"):
                (tmp/d).mkdir(parents=True, exist_ok=True)

            symlink_force(DB_CSV, tmp/"db.csv")
            symlink_force(src["video"], tmp/"video"/src["video"].name)
            symlink_force(src["ppg_raw"], tmp/"ppg"/src["ppg_raw"].name)
            symlink_force(src["ecg"], tmp/"ecg"/src["ecg"].name)

            # Exact prefix only; no interpolation, duplication or synthetic frame.
            (tmp/"meta"/src["meta"].name).write_text(
                "".join(meta_lines[:n_usable]), encoding="utf-8"
            )
            (tmp/"ppg_sync"/src["ppg_sync"].name).write_text(
                "".join(sync_lines[:n_usable]), encoding="utf-8"
            )

            args = SimpleNamespace(
                data_root=tmp,
                output_root=CLIP_ROOT,
                predictor=PREDICTOR_PATH,
                clip_len=CLIP_LEN,
                stride=STRIDE,
                roi_size=ROI_SIZE,
                gzip_level=GZIP_LEVEL,
                min_free_gb=MIN_FREE_GB,
                rgb_store=False,
                overwrite=False,
                skip_validation=False,
            )

            info = mod.process_recording(
                subject, state, camera, args, db, extractor
            )
            mod.validate_random_clips(Path(info["h5_path"]), n_checks=8)

            patch_provenance_and_counts(
                Path(info["h5_path"]),
                done_path,
                src,
                n_meta_orig,
                n_sync_orig,
                n_decodable,
                n_usable,
            )

            # Revalidate after patch.
            mod.validate_random_clips(Path(info["h5_path"]), n_checks=8)
            audit = structural_h5_audit(Path(info["h5_path"]))
            if not audit["structural_ok"]:
                raise RuntimeError("Post-repair HDF5 structural audit failed.")

            row["status"] = "REPAIRED"
            row["h5_path"] = str(info["h5_path"])
            row["n_clips"] = audit["n_clips"]
            row["fallback_ratio"] = audit["fallback_ratio"]
            row["sync_mean_s"] = audit["sync_mean_s"]
            row["sync_max_s"] = audit["sync_max_s"]
            print(
                f'  REPAIRED | usable={n_usable} | '
                f'shortfall={row["decode_shortfall_vs_old_rule"]} frame(s) | '
                f'clips={audit["n_clips"]} | fallback={100*audit["fallback_ratio"]:.2f}%'
            )

            shutil.rmtree(tmp, ignore_errors=True)

        except Exception as e:
            row["status"] = "ERROR"
            row["repair_error"] = repr(e)
            print("  ERROR:", repr(e))

        report.append(row)
        write_csv(RESULTS/"01_decode_repair_report.csv", report)

    summary = {
        "queue": len(report),
        "repaired": sum(r["status"]=="REPAIRED" for r in report),
        "already_complete": sum(r["status"]=="ALREADY_COMPLETE" for r in report),
        "blocked_or_error": sum(r["status"] not in ("REPAIRED","ALREADY_COMPLETE") for r in report),
        "repair_rule": "n_usable=min(n_meta,n_sync,n_actual_decodable)",
        "raw_source_files_modified": False,
        "exact_original_roi_extractor_reused": True,
    }
    save_json(RESULTS/"01_decode_repair_summary.json", summary)
    print("\n" + json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
