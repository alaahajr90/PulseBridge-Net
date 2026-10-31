from __future__ import annotations
import csv, json, os, hashlib, importlib.util, shutil
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np

from config_qcv2 import *


def ensure_dirs():
    for p in (RESULTS, LOGS, BACKUPS, TEMP_ROOT):
        p.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def sid(x):
    try:
        return str(int(float(str(x).strip())))
    except Exception:
        return str(x).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(path="/"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def guard_disk():
    x = free_gb("/")
    if x < MIN_FREE_GB:
        raise RuntimeError(f"Free disk {x:.2f} GiB below safety threshold {MIN_FREE_GB:.1f} GiB")
    return x


def load_extractor_module():
    if not EXTRACTOR_PATH.exists():
        raise FileNotFoundError(EXTRACTOR_PATH)
    spec = importlib.util.spec_from_file_location("mcd_compact_original", EXTRACTOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def structural_h5_audit(path):
    with h5py.File(path, "r") as h:
        required = [
            "roi", "ppg", "ppg_sync_dt", "coords", "frame_ids",
            "frame_timestamps", "face_detected", "fallback_used",
            "clip_starts", "labels",
        ]
        missing = [k for k in required if k not in h]
        if missing:
            return {"structural_ok": False, "missing_keys": ";".join(missing)}

        n = len(h["roi"])
        aligned = all(len(h[k]) == n for k in [
            "ppg","ppg_sync_dt","coords","frame_ids","frame_timestamps",
            "face_detected","fallback_used"
        ])
        roi_shape = h["roi"].ndim == 4 and tuple(h["roi"].shape[1:]) == (ROI_SIZE, ROI_SIZE, 3)
        starts = h["clip_starts"][:]
        starts_ok = len(starts) > 0 and np.all(starts >= 0) and np.all(starts + CLIP_LEN <= n)
        finite = np.isfinite(h["ppg"][:]).all() and np.isfinite(h["ppg_sync_dt"][:]).all()

        canonical = h["labels"]["canonical"].attrs if "canonical" in h["labels"] else {}
        labels = {}
        for name in STAGE3_LABELS:
            val = canonical.get(name, np.nan)
            try:
                labels[name] = float(val)
            except Exception:
                labels[name] = np.nan

        return {
            "structural_ok": bool(aligned and roi_shape and starts_ok and finite),
            "missing_keys": "",
            "aligned_lengths_ok": bool(aligned),
            "roi_shape_ok": bool(roi_shape),
            "clip_starts_ok": bool(starts_ok),
            "finite_ppg_sync_ok": bool(finite),
            "subject_id": str(h.attrs.get("subject_id","")),
            "state": str(h.attrs.get("state","")),
            "camera": str(h.attrs.get("camera","")),
            "view": str(h.attrs.get("view","")),
            "n_usable": int(h.attrs.get("n_usable", n)),
            "n_meta": int(h.attrs.get("n_meta", n)),
            "n_sync": int(h.attrs.get("n_sync", n)),
            "n_clips": int(h.attrs.get("n_clips", len(starts))),
            "fallback_ratio": float(h.attrs.get("fallback_ratio", np.mean(h["fallback_used"][:]))),
            "sync_mean_s": float(np.mean(h["ppg_sync_dt"][:])),
            "sync_max_s": float(np.max(h["ppg_sync_dt"][:])),
            "systolic_bp": labels["systolic_bp"],
            "diastolic_bp": labels["diastolic_bp"],
            "rigidity": labels["rigidity"],
            "h5_path": str(path),
        }


def sync_class(mean_dt, max_dt):
    if mean_dt <= SYNC_GOOD_MEAN_MAX_S and max_dt <= SYNC_GOOD_MAX_MAX_S:
        return "GOOD"
    if mean_dt <= SYNC_REVIEW_MEAN_MAX_S and max_dt <= SYNC_REVIEW_MAX_MAX_S:
        return "REVIEW"
    return "BAD"
