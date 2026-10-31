#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCD-rPPG compact / non-redundant clip extractor for the HR/RR project.

Scientific clip specification (kept identical to the previous project):
    CLIP_LEN = 128 frames
    STRIDE   = 8 frames
    ROI_SIZE = 64 x 64
    COLOR    = BGR (legacy-compatible)

Why this version saves disk space
---------------------------------
The old extractor physically saved the 128 ROI frames inside every overlapping
clip. With stride=8, most frames were therefore stored ~16 times.

This extractor stores EACH RECORDING ONLY ONCE in one HDF5 file:
    roi[N,64,64,3]
    ppg[N]
    ppg_sync_dt[N]
    coords[N,4]
    frame_ids[N]
    frame_timestamps[N]
    face_detected[N]
    fallback_used[N]
    clip_starts[K]

A logical clip k is reconstructed exactly as:
    start = clip_starts[k]
    end   = start + 128
    roi_clip = roi[start:end]
    ppg_clip = ppg[start:end]

Therefore the MODEL receives the same 128-frame clip; only on-disk storage is
changed. All db.csv columns are preserved once per recording and are inherited
by every logical clip from that recording.

Safety
------
- Source dataset is READ-ONLY.
- Writes only under OUTPUT_ROOT.
- Uses temporary .tmp.h5 files and atomically renames on successful completion.
- Resume-safe: completed recordings have _DONE.json and are skipped unless
  --overwrite is supplied.
- Stops before disk becomes critically full (default: < 20 GiB free).

Default run: Subject 1020 only.
Use --all_subjects to process all 600 subjects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DATA_ROOT_DEFAULT = Path("/home/hajr/mcd_rppg")
OUTPUT_ROOT_DEFAULT = Path("/home/hajr/Desktop/mcd_rppg_clips")
PREDICTOR_DEFAULT = Path(
    "/home/hajr/alaa_hajr/shape_predictor_68_face_landmarks.dat"
)

CLIP_LEN_DEFAULT = 128
STRIDE_DEFAULT = 8
ROI_SIZE_DEFAULT = 64
GZIP_LEVEL_DEFAULT = 4
MIN_FREE_GB_DEFAULT = 20.0

CAMERAS = ["FullHDwebcam", "USBVideo", "IriunWebcam"]
STATES = ["before", "after"]
CAMERA_TO_VIEW = {
    "FullHDwebcam": "front",
    "USBVideo": "left",
    "IriunWebcam": "right",
}

SCHEMA_VERSION = "mcd_rppg_compact_v2.0"
RECORDING_FILENAME = "recording.h5"
DONE_FILENAME = "_DONE.json"

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
try:
    cv2.setLogLevel(0)
except Exception:
    pass


# =============================================================================
# GENERIC HELPERS
# =============================================================================
def norm_text(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def safe_key(name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip())
    key = re.sub(r"_+", "_", key).strip("_")
    if not key:
        key = "unnamed"
    if key[0].isdigit():
        key = "col_" + key
    return key


def parse_timestamp(text: str) -> datetime:
    text = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp format: {text!r}")


def effective_fps(times: List[datetime]) -> float:
    if len(times) < 2:
        return float("nan")
    duration = (times[-1] - times[0]).total_seconds()
    if duration <= 0:
        return float("nan")
    return float((len(times) - 1) / duration)


def relative_seconds(times: List[datetime]) -> np.ndarray:
    if not times:
        return np.empty((0,), dtype=np.float64)
    t0 = times[0]
    return np.asarray(
        [(t - t0).total_seconds() for t in times],
        dtype=np.float64,
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def free_space_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def require_inside_output(path: Path, output_root: Path) -> None:
    p = path.resolve()
    root = output_root.resolve()
    if p != root and root not in p.parents:
        raise RuntimeError(f"Safety error: output path escapes OUTPUT_ROOT: {p}")


# =============================================================================
# db.csv
# =============================================================================
class DBTable:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise RuntimeError(f"db.csv has no header: {path}")
            self.columns = list(reader.fieldnames)
            self.rows = [dict(r) for r in reader]

        if "patient_id" not in self.columns:
            raise RuntimeError("db.csv must contain patient_id")

        self.by_subject: Dict[str, List[Dict[str, str]]] = {}
        for row in self.rows:
            sid = self._sid(row.get("patient_id", ""))
            self.by_subject.setdefault(sid, []).append(row)

    @staticmethod
    def _sid(x) -> str:
        s = str(x).strip()
        try:
            return str(int(float(s)))
        except Exception:
            return s

    def subjects(self) -> List[str]:
        return sorted(
            self.by_subject.keys(),
            key=lambda x: int(x) if x.isdigit() else x,
        )

    def find_row(
        self,
        sid: str,
        state: str,
        camera: str,
        video_path: Path,
    ) -> Dict[str, str]:
        candidates = list(self.by_subject.get(str(sid), []))
        if not candidates:
            raise KeyError(f"No db.csv rows for subject {sid}")

        def filter_if_informative(rows, col, target):
            if col not in self.columns:
                return rows
            nonempty = [r for r in rows if str(r.get(col, "")).strip()]
            if not nonempty:
                return rows
            matched = [
                r for r in nonempty
                if norm_text(r.get(col, "")) == norm_text(target)
            ]
            return matched if matched else rows

        candidates = filter_if_informative(candidates, "step", state)
        candidates = filter_if_informative(candidates, "camera", camera)

        if len(candidates) > 1 and "video" in self.columns:
            base = norm_text(video_path.name)
            matched = [
                r for r in candidates
                if base and base in norm_text(r.get("video", ""))
            ]
            if matched:
                candidates = matched

        if len(candidates) > 1 and "view" in self.columns:
            wanted_view = CAMERA_TO_VIEW.get(camera, "")
            matched = [
                r for r in candidates
                if norm_text(r.get("view", "")) == norm_text(wanted_view)
            ]
            if matched:
                candidates = matched

        if len(candidates) != 1:
            brief = [
                {
                    "patient_id": r.get("patient_id", ""),
                    "step": r.get("step", ""),
                    "camera": r.get("camera", ""),
                    "view": r.get("view", ""),
                    "video": r.get("video", ""),
                }
                for r in candidates[:10]
            ]
            raise RuntimeError(
                "Ambiguous db.csv row: "
                f"subject={sid}, state={state}, camera={camera}, "
                f"candidates={len(candidates)}, sample={brief}"
            )

        return candidates[0]


# =============================================================================
# INPUT LOADERS
# =============================================================================
def load_meta(path: Path) -> Tuple[np.ndarray, List[datetime], List[str]]:
    frame_ids: List[int] = []
    times: List[datetime] = []
    raw_times: List[str] = []

    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Bad meta line {line_no} in {path}: {line!r}"
                )
            frame_ids.append(int(parts[0]))
            raw_times.append(parts[1].strip())
            times.append(parse_timestamp(parts[1]))

    return (
        np.asarray(frame_ids, dtype=np.int32),
        times,
        raw_times,
    )


def load_ppg_sync(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ppg: List[float] = []
    dt: List[float] = []

    with open(path, "r", encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Bad ppg_sync line {line_no} in {path}: {line!r}"
                )
            ppg.append(float(parts[0]))
            dt.append(float(parts[1]))

    return (
        np.asarray(ppg, dtype=np.float32),
        np.asarray(dt, dtype=np.float32),
    )


# =============================================================================
# ROI EXTRACTION -- IDENTICAL LEGACY GEOMETRY
# =============================================================================
class ROIExtractor:
    def __init__(
        self,
        predictor_path: Path,
        roi_size: int,
        rgb_store: bool = False,
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.roi_size = int(roi_size)
        self.rgb_store = bool(rgb_store)

    def _crop_resize(
        self,
        frame: np.ndarray,
        coords: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x_min, x_max, y_min, y_max = coords
        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            roi = frame[
                max(cy - 32, 0):min(cy + 32, h),
                max(cx - 32, 0):min(cx + 32, w),
            ]

        if roi.size == 0:
            raise RuntimeError("ROI empty even after emergency center crop")

        roi = cv2.resize(
            roi,
            (self.roi_size, self.roi_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.rgb_store:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        return roi

    def extract_with_fallback(
        self,
        frame: np.ndarray,
        prev_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool, bool]:
        """
        Exact legacy ROI geometry from the previous notebook:
            x range: landmarks 17:27 and 1:16
            y top  : min landmarks 19:24 minus 40 pixels
            y bottom: landmark 33 plus 30 pixels

        If face detection fails, previous frame coordinates are reused.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        face_detected = len(faces) > 0
        fallback_used = not face_detected

        if not face_detected:
            if prev_coords is None:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                coords = (
                    max(cx - 32, 0),
                    min(cx + 32, w),
                    max(cy - 32, 0),
                    min(cy + 32, h),
                )
            else:
                coords = prev_coords
        else:
            # Preserve legacy behavior: first detected face.
            shape = self.predictor(gray, faces[0])
            pts = np.asarray(
                [(p.x, p.y) for p in shape.parts()],
                dtype=np.int32,
            )

            x_min = min(
                pts[17:27, 0].min(),
                pts[1:16, 0].min(),
            )
            x_max = max(
                pts[17:27, 0].max(),
                pts[1:16, 0].max(),
            )
            y_forehead_top = pts[19:24, 1].min()
            y_nose_base = pts[33, 1]

            h, w = frame.shape[:2]
            x_min = max(int(x_min), 0)
            x_max = min(int(x_max), w)
            y_min = max(int(y_forehead_top - 40), 0)
            y_max = min(int(y_nose_base + 30), h)
            coords = (x_min, x_max, y_min, y_max)

        roi = self._crop_resize(frame, coords)
        return roi, coords, face_detected, fallback_used


# =============================================================================
# VIDEO ROI EXTRACTION
# =============================================================================
def extract_recording_rois(
    video_path: Path,
    n_usable: int,
    extractor: ROIExtractor,
    roi_size: int,
):
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap_frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    rois = np.empty(
        (n_usable, roi_size, roi_size, 3),
        dtype=np.uint8,
    )
    coords = np.empty((n_usable, 4), dtype=np.int32)
    face_detected = np.zeros((n_usable,), dtype=np.uint8)
    fallback_used = np.zeros((n_usable,), dtype=np.uint8)

    prev_coords = None
    decoded = 0

    try:
        for i in tqdm(
            range(n_usable),
            desc=f"ROI {video_path.stem}",
            leave=False,
        ):
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Decode stopped at frame {i}; "
                    f"expected n_usable={n_usable}: {video_path}"
                )

            roi, c, detected, fallback = extractor.extract_with_fallback(
                frame,
                prev_coords,
            )

            rois[i] = roi
            coords[i] = np.asarray(c, dtype=np.int32)
            face_detected[i] = 1 if detected else 0
            fallback_used[i] = 1 if fallback else 0
            prev_coords = c
            decoded += 1

    finally:
        cap.release()

    return {
        "rois": rois,
        "coords": coords,
        "face_detected": face_detected,
        "fallback_used": fallback_used,
        "video_fps": video_fps,
        "cap_frame_count": cap_frame_count,
        "decoded_frames": decoded,
    }


# =============================================================================
# PATHS / RECORDING ID
# =============================================================================
def required_paths(
    root: Path,
    sid: str,
    state: str,
    camera: str,
) -> Dict[str, Path]:
    return {
        "video": root / "video" / f"{sid}_{camera}_{state}.avi",
        "meta": root / "meta" / f"{sid}_{camera}_{state}.txt",
        "ppg_sync": root / "ppg_sync" / f"{sid}_{camera}_{state}.txt",
        "ppg_raw": root / "ppg" / f"{sid}_{state}.PW",
        "ecg": root / "ecg" / f"{sid}_{state}.json",
    }


def validate_source_paths(paths: Dict[str, Path]) -> None:
    missing = [f"{k}: {p}" for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing source files:\n  " + "\n  ".join(missing)
        )


def recording_output_dir(
    output_root: Path,
    sid: str,
    camera: str,
    state: str,
) -> Path:
    return output_root / f"subject_{sid}" / camera / state


def recording_uid(sid: str, camera: str, state: str) -> str:
    return f"{sid}_{camera}_{state}"


# =============================================================================
# HDF5 WRITING
# =============================================================================
def string_array(values: List[str]):
    dt = h5py.string_dtype(encoding="utf-8")
    return np.asarray([str(v) for v in values], dtype=object), dt


def write_db_metadata(
    h5: h5py.File,
    db_row: Dict[str, str],
    db_columns: List[str],
) -> None:
    """
    Preserve every db.csv column losslessly ONCE per recording.
    Every logical clip inherits these labels from its recording.
    """
    g = h5.create_group("labels")
    utf8 = h5py.string_dtype(encoding="utf-8")

    cols = [str(c) for c in db_columns]
    vals = ["" if db_row.get(c) is None else str(db_row.get(c, "")) for c in cols]

    g.create_dataset("db_columns", data=np.asarray(cols, dtype=object), dtype=utf8)
    g.create_dataset("db_values", data=np.asarray(vals, dtype=object), dtype=utf8)

    db_json = json.dumps(
        {c: db_row.get(c, "") for c in cols},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    g.attrs["db_row_json"] = db_json

    # Convenient canonical aliases for future tasks.
    aliases = {
        "patient_id": "patient_id",
        "weight": "weight_kg",
        "height": "height_cm",
        "bmi": "bmi",
        "age": "age_years",
        "sex": "sex",
        "upper_ap": "systolic_bp",
        "lower_ap": "diastolic_bp",
        "saturation": "spo2",
        "temperature": "temperature_c",
        "hemoglobin": "hemoglobin",
        "glycated_hemoglobin": "hba1c",
        "cholesterol": "cholesterol",
        "respiratory": "respiratory_rate",
        "rigidity": "rigidity",
        "pulse": "pulse_bpm",
        "stress": "stress",
        "step": "db_step",
        "camera": "db_camera",
        "view": "db_view",
        "ecg": "db_ecg",
        "ppg": "db_ppg",
        "video": "db_video",
        "meta": "db_meta",
        "ppg_sync": "db_ppg_sync",
    }

    canonical = g.create_group("canonical")
    for src, dst in aliases.items():
        if src not in db_row:
            continue
        raw = "" if db_row.get(src) is None else str(db_row.get(src, "")).strip()
        # Preserve raw text always.
        canonical.attrs[dst + "__raw"] = raw
        # Numeric version when possible.
        try:
            if raw != "":
                canonical.attrs[dst] = float(raw)
            else:
                canonical.attrs[dst] = ""
        except Exception:
            canonical.attrs[dst] = raw

    # Also expose every original column as an attribute under original/.
    original = g.create_group("original")
    used = set()
    for col in cols:
        key = safe_key(col)
        if key in used:
            suffix = 2
            while f"{key}_{suffix}" in used:
                suffix += 1
            key = f"{key}_{suffix}"
        used.add(key)
        original.attrs[key] = "" if db_row.get(col) is None else str(db_row.get(col, ""))
        original.attrs[key + "__source_column"] = col


# =============================================================================
# PROCESS ONE RECORDING
# =============================================================================
def process_recording(
    sid: str,
    state: str,
    camera: str,
    args,
    db: DBTable,
    extractor: ROIExtractor,
) -> Dict[str, object]:
    src = required_paths(args.data_root, sid, state, camera)
    validate_source_paths(src)

    out_dir = recording_output_dir(args.output_root, sid, camera, state)
    require_inside_output(out_dir, args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / RECORDING_FILENAME
    tmp_h5_path = out_dir / (RECORDING_FILENAME + ".tmp")
    done_path = out_dir / DONE_FILENAME

    # Resume.
    if done_path.exists() and h5_path.exists() and not args.overwrite:
        info = json.loads(done_path.read_text(encoding="utf-8"))
        print(
            f"[SKIP completed] {sid} | {camera} | {state} | "
            f"clips={info.get('n_clips')} | "
            f"size={info.get('h5_size_mb', 0):.1f} MB"
        )
        return info

    # Free-space guard BEFORE expensive work.
    free_gb = free_space_gb(args.output_root)
    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"LOW DISK SPACE: {free_gb:.2f} GiB free < "
            f"minimum {args.min_free_gb:.2f} GiB. Stopping safely."
        )

    # Remove only temporary output from an interrupted run.
    if tmp_h5_path.exists():
        tmp_h5_path.unlink()

    db_row = db.find_row(sid, state, camera, src["video"])

    # Load exact timing/sync metadata.
    frame_ids_all, meta_times_all, meta_time_strings_all = load_meta(src["meta"])
    ppg_all, ppg_dt_all = load_ppg_sync(src["ppg_sync"])

    if len(ppg_all) != len(ppg_dt_all):
        raise RuntimeError("ppg_sync amplitude/dt length mismatch")
    if len(ppg_all) > len(frame_ids_all):
        raise RuntimeError(
            f"ppg_sync rows={len(ppg_all)} > meta rows={len(frame_ids_all)}"
        )

    # ppg_sync defines the synchronized usable prefix established by our audit.
    n_meta = len(frame_ids_all)
    n_sync = len(ppg_all)
    n_usable = min(n_meta, n_sync)

    if n_usable < args.clip_len:
        raise RuntimeError(
            f"n_usable={n_usable} < clip_len={args.clip_len}"
        )

    frame_ids = frame_ids_all[:n_usable]
    meta_times = meta_times_all[:n_usable]
    meta_time_strings = meta_time_strings_all[:n_usable]
    ppg = ppg_all[:n_usable]
    ppg_dt = ppg_dt_all[:n_usable]

    starts = np.arange(
        0,
        n_usable - args.clip_len + 1,
        args.stride,
        dtype=np.int32,
    )
    n_clips = int(len(starts))

    if n_clips <= 0:
        raise RuntimeError("No valid clips generated")

    # Extract each usable ROI frame ONCE.
    rec = extract_recording_rois(
        src["video"],
        n_usable,
        extractor,
        args.roi_size,
    )

    rois = rec["rois"]
    coords = rec["coords"]
    face_detected = rec["face_detected"]
    fallback_used = rec["fallback_used"]

    if rois.shape != (n_usable, args.roi_size, args.roi_size, 3):
        raise RuntimeError(f"Unexpected ROI array shape: {rois.shape}")

    if not np.isfinite(ppg).all() or not np.isfinite(ppg_dt).all():
        raise RuntimeError("NaN/Inf in ppg_sync")

    rel_recording = relative_seconds(meta_times)
    meta_fps = effective_fps(meta_times)

    # Frame timestamps are preserved exactly as source strings.
    utf8 = h5py.string_dtype(encoding="utf-8")

    # Chunk along time. 128 aligns well with the scientific clip length.
    roi_chunk_t = min(max(args.clip_len, 1), n_usable)
    roi_chunks = (roi_chunk_t, args.roi_size, args.roi_size, 3)

    # Write to temporary HDF5 first.
    with h5py.File(tmp_h5_path, "w") as h5:
        # --------------------------- identity / provenance --------------------
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["subject_id"] = str(sid)
        h5.attrs["state"] = state
        h5.attrs["camera"] = camera
        h5.attrs["view"] = CAMERA_TO_VIEW[camera]
        h5.attrs["color_space"] = "RGB" if args.rgb_store else "BGR"
        h5.attrs["clip_len"] = int(args.clip_len)
        h5.attrs["stride"] = int(args.stride)
        h5.attrs["roi_size"] = int(args.roi_size)
        h5.attrs["n_usable"] = int(n_usable)
        h5.attrs["n_meta"] = int(n_meta)
        h5.attrs["n_sync"] = int(n_sync)
        h5.attrs["tail_frames"] = int(n_meta - n_usable)
        h5.attrs["n_clips"] = int(n_clips)
        h5.attrs["video_fps_reported"] = float(rec["video_fps"])
        h5.attrs["meta_fps"] = float(meta_fps) if math.isfinite(meta_fps) else np.nan
        h5.attrs["video_cap_frame_count"] = int(rec["cap_frame_count"])
        h5.attrs["decoded_frames"] = int(rec["decoded_frames"])
        h5.attrs["fallback_ratio"] = float(fallback_used.mean())
        h5.attrs["face_detection_ratio"] = float(face_detected.mean())
        h5.attrs["ppg_sync_dt_mean_s"] = float(ppg_dt.mean())
        h5.attrs["ppg_sync_dt_max_s"] = float(ppg_dt.max())
        h5.attrs["ppg_sync_dt_min_s"] = float(ppg_dt.min())
        h5.attrs["source_video"] = str(src["video"])
        h5.attrs["source_meta"] = str(src["meta"])
        h5.attrs["source_ppg_sync"] = str(src["ppg_sync"])
        h5.attrs["source_ppg_raw"] = str(src["ppg_raw"])
        h5.attrs["source_ecg"] = str(src["ecg"])
        h5.attrs["source_db_csv"] = str(db.path)

        # ------------------------------- core arrays --------------------------
        h5.create_dataset(
            "roi",
            data=rois,
            dtype=np.uint8,
            chunks=roi_chunks,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )

        # Signals/timing are tiny; also compress them.
        h5.create_dataset(
            "ppg",
            data=ppg.astype(np.float32, copy=False),
            dtype=np.float32,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        # Alias is NOT duplicated physically; consumers should use ppg.
        # Attribute records the legacy correspondence instead.
        h5.attrs["legacy_bvp_dataset"] = "ppg"

        h5.create_dataset(
            "ppg_sync_dt",
            data=ppg_dt.astype(np.float32, copy=False),
            dtype=np.float32,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        h5.create_dataset(
            "coords",
            data=coords.astype(np.int32, copy=False),
            dtype=np.int32,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        h5.create_dataset(
            "frame_ids",
            data=frame_ids.astype(np.int32, copy=False),
            dtype=np.int32,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        h5.create_dataset(
            "frame_time_from_recording_start_s",
            data=rel_recording.astype(np.float64, copy=False),
            dtype=np.float64,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        h5.create_dataset(
            "frame_timestamps",
            data=np.asarray(meta_time_strings, dtype=object),
            dtype=utf8,
            compression="gzip",
            compression_opts=args.gzip_level,
        )
        h5.create_dataset(
            "face_detected",
            data=face_detected.astype(np.uint8, copy=False),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=args.gzip_level,
        )
        h5.create_dataset(
            "fallback_used",
            data=fallback_used.astype(np.uint8, copy=False),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=args.gzip_level,
        )

        # ------------------------- logical clip definitions ------------------
        # This is the key to non-redundant storage.
        h5.create_dataset(
            "clip_starts",
            data=starts,
            dtype=np.int32,
            compression="gzip",
            compression_opts=args.gzip_level,
            shuffle=True,
        )
        h5.attrs["clip_end_rule"] = "end_exclusive = start + clip_len"

        # ---------------------------- every db.csv label ---------------------
        write_db_metadata(h5, db_row, db.columns)

        h5.flush()

    # Atomic commit.
    os.replace(tmp_h5_path, h5_path)

    h5_mb = file_size_mb(h5_path)
    free_after_gb = free_space_gb(args.output_root)

    info = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": str(sid),
        "state": state,
        "camera": camera,
        "view": CAMERA_TO_VIEW[camera],
        "n_meta": int(n_meta),
        "n_sync": int(n_sync),
        "n_usable": int(n_usable),
        "tail_frames": int(n_meta - n_usable),
        "n_clips": int(n_clips),
        "clip_len": int(args.clip_len),
        "stride": int(args.stride),
        "roi_size": int(args.roi_size),
        "video_fps": float(rec["video_fps"]),
        "meta_fps": float(meta_fps) if math.isfinite(meta_fps) else None,
        "fallback_ratio": float(fallback_used.mean()),
        "face_detection_ratio": float(face_detected.mean()),
        "ppg_sync_dt_mean_s": float(ppg_dt.mean()),
        "ppg_sync_dt_max_s": float(ppg_dt.max()),
        "h5_path": str(h5_path),
        "h5_size_mb": float(h5_mb),
        "free_disk_after_gb": float(free_after_gb),
        "status": "OK",
    }

    atomic_write_text(
        done_path,
        json.dumps(info, indent=2, ensure_ascii=False),
    )

    return info


# =============================================================================
# VALIDATE A LOGICAL CLIP FROM HDF5
# =============================================================================
def validate_random_clips(h5_path: Path, n_checks: int = 5) -> None:
    """Quick structural verification; no source data are modified."""
    rng = np.random.default_rng(42)

    with h5py.File(h5_path, "r") as h5:
        clip_len = int(h5.attrs["clip_len"])
        starts = h5["clip_starts"][:]

        if len(starts) == 0:
            raise RuntimeError(f"No clip_starts in {h5_path}")

        ids = rng.choice(
            len(starts),
            size=min(n_checks, len(starts)),
            replace=False,
        )

        for k in ids:
            start = int(starts[k])
            end = start + clip_len

            roi = h5["roi"][start:end]
            ppg = h5["ppg"][start:end]
            dt = h5["ppg_sync_dt"][start:end]
            coords = h5["coords"][start:end]
            frame_ids = h5["frame_ids"][start:end]

            roi_size = int(h5.attrs["roi_size"])
            if roi.shape != (clip_len, roi_size, roi_size, 3):
                raise RuntimeError(
                    f"Bad logical clip ROI shape {roi.shape} in {h5_path}"
                )
            if not (
                len(ppg) == len(dt) == len(coords) == len(frame_ids) == clip_len
            ):
                raise RuntimeError(
                    f"Logical clip alignment mismatch in {h5_path}"
                )
            if not np.isfinite(ppg).all() or not np.isfinite(dt).all():
                raise RuntimeError(f"NaN/Inf in logical clip: {h5_path}")


# =============================================================================
# CSV / SUMMARY
# =============================================================================
SUMMARY_FIELDS = [
    "subject_id",
    "state",
    "camera",
    "view",
    "status",
    "n_meta",
    "n_sync",
    "n_usable",
    "tail_frames",
    "n_clips",
    "clip_len",
    "stride",
    "roi_size",
    "video_fps",
    "meta_fps",
    "fallback_ratio",
    "face_detection_ratio",
    "ppg_sync_dt_mean_s",
    "ppg_sync_dt_max_s",
    "h5_path",
    "h5_size_mb",
    "free_disk_after_gb",
    "error",
]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

    os.replace(tmp, path)


# =============================================================================
# COMMAND LINE
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Compact non-redundant MCD-rPPG clip extractor"
    )

    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT_DEFAULT,
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--predictor",
        type=Path,
        default=PREDICTOR_DEFAULT,
    )

    p.add_argument("--clip-len", type=int, default=CLIP_LEN_DEFAULT)
    p.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    p.add_argument("--roi-size", type=int, default=ROI_SIZE_DEFAULT)
    p.add_argument(
        "--gzip-level",
        type=int,
        default=GZIP_LEVEL_DEFAULT,
        choices=range(0, 10),
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB_DEFAULT,
        help="Stop safely if available disk drops below this value",
    )

    p.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subject IDs, e.g. --subjects 1020 1287",
    )
    p.add_argument(
        "--all_subjects",
        action="store_true",
        help="Process all 600 subjects",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild completed recording files",
    )
    p.add_argument(
        "--rgb-store",
        action="store_true",
        help="Store ROI as RGB. Default BGR preserves legacy behavior.",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip random logical-clip validation after each recording",
    )

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.predictor = args.predictor.expanduser().resolve()

    if args.clip_len <= 0 or args.stride <= 0 or args.roi_size <= 0:
        raise ValueError("clip_len, stride, roi_size must be positive")

    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    if not args.predictor.exists():
        raise FileNotFoundError(args.predictor)

    db_path = args.data_root / "db.csv"
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Never allow the output root to be one of the scientific source folders.
    protected = {
        (args.data_root / "video").resolve(),
        (args.data_root / "ppg").resolve(),
        (args.data_root / "ecg").resolve(),
        (args.data_root / "meta").resolve(),
        (args.data_root / "ppg_sync").resolve(),
    }
    if args.output_root in protected:
        raise RuntimeError("OUTPUT_ROOT must not overlap source scientific folders")

    db = DBTable(db_path)

    if args.all_subjects and args.subjects:
        raise ValueError("Use either --all_subjects OR --subjects, not both")

    if args.all_subjects:
        subjects = db.subjects()
    elif args.subjects:
        subjects = [str(x) for x in args.subjects]
    else:
        subjects = ["1020"]

    extractor = ROIExtractor(
        args.predictor,
        args.roi_size,
        rgb_store=args.rgb_store,
    )

    print("=" * 108)
    print("MCD-rPPG COMPACT / NON-REDUNDANT CLIP EXTRACTION")
    print("=" * 108)
    print(f"DATA_ROOT      : {args.data_root}")
    print(f"OUTPUT_ROOT    : {args.output_root}")
    print(f"PREDICTOR      : {args.predictor}")
    print(f"CLIP_LEN       : {args.clip_len}")
    print(f"STRIDE         : {args.stride}")
    print(f"ROI_SIZE       : {args.roi_size}")
    print(f"COLOR          : {'RGB' if args.rgb_store else 'BGR'}")
    print(f"HDF5 gzip      : level {args.gzip_level}")
    print(f"MIN FREE DISK  : {args.min_free_gb:.1f} GiB")
    print(f"Free disk now  : {free_space_gb(args.output_root):.2f} GiB")
    print()
    print(f"db.csv         : {db_path}")
    print(f"db rows        : {len(db.rows)}")
    print(f"db columns     : {len(db.columns)} -> {db.columns}")
    print(f"Subjects       : {len(subjects)}")
    if len(subjects) <= 50:
        print(f"IDs            : {', '.join(subjects)}")
    print()
    print(
        "Storage rule  : each recording stored ONCE; "
        "clips are slices defined by clip_starts"
    )
    print(
        "Logical clip  : roi[start:start+128], ppg[start:start+128], "
        "same labels from recording"
    )

    manifests_dir = args.output_root / "_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    total_recordings = len(subjects) * len(STATES) * len(CAMERAS)
    counter = 0
    total_logical_clips = 0
    total_h5_gb = 0.0

    for sid in subjects:
        for state in STATES:
            for camera in CAMERAS:
                counter += 1
                print("\n" + "-" * 108)
                print(
                    f"[{counter:04d}/{total_recordings:04d}] "
                    f"subject={sid} | state={state} | camera={camera} | "
                    f"free={free_space_gb(args.output_root):.2f} GiB"
                )

                try:
                    info = process_recording(
                        sid,
                        state,
                        camera,
                        args,
                        db,
                        extractor,
                    )

                    if not args.skip_validation:
                        validate_random_clips(
                            Path(info["h5_path"]),
                            n_checks=5,
                        )

                    info["error"] = ""
                    results.append(info)

                    total_logical_clips += int(info["n_clips"])
                    total_h5_gb += float(info["h5_size_mb"]) / 1024.0

                    print(
                        f"[OK] clips={info['n_clips']} | "
                        f"usable={info['n_usable']} | "
                        f"tail={info['tail_frames']} | "
                        f"fallback={100*float(info['fallback_ratio']):.2f}% | "
                        f"H5={float(info['h5_size_mb']):.1f} MB | "
                        f"free={float(info['free_disk_after_gb']):.2f} GiB"
                    )

                except Exception as e:
                    err = {
                        "subject_id": str(sid),
                        "state": state,
                        "camera": camera,
                        "view": CAMERA_TO_VIEW.get(camera, ""),
                        "status": "ERROR",
                        "error": repr(e),
                    }
                    errors.append(err)
                    results.append(err)
                    print(f"[ERROR] {repr(e)}")

                    # A low-disk event should stop the full run immediately.
                    if "LOW DISK SPACE" in str(e):
                        print("\nSTOPPING SAFELY because free disk reached the configured minimum.")
                        write_csv(manifests_dir / "recordings_summary.csv", results)
                        write_csv(manifests_dir / "errors.csv", errors)
                        sys.exit(2)

                # Rewrite summary after every recording so progress survives interruption.
                write_csv(manifests_dir / "recordings_summary.csv", results)
                write_csv(manifests_dir / "errors.csv", errors)

                run_summary = {
                    "schema_version": SCHEMA_VERSION,
                    "subjects_requested": len(subjects),
                    "recordings_processed_or_skipped": sum(
                        1 for r in results if r.get("status") == "OK"
                    ),
                    "recordings_errors": len(errors),
                    "total_logical_clips": int(total_logical_clips),
                    "total_h5_gb_current_run": float(total_h5_gb),
                    "free_disk_gb": float(free_space_gb(args.output_root)),
                    "clip_len": int(args.clip_len),
                    "stride": int(args.stride),
                    "roi_size": int(args.roi_size),
                    "output_root": str(args.output_root),
                }
                atomic_write_text(
                    manifests_dir / "run_summary.json",
                    json.dumps(run_summary, indent=2, ensure_ascii=False),
                )

    ok_count = sum(1 for r in results if r.get("status") == "OK")

    print("\n" + "=" * 108)
    print("COMPACT EXTRACTION FINISHED")
    print("=" * 108)
    print(f"Recordings OK/skipped : {ok_count}/{total_recordings}")
    print(f"Recordings errors     : {len(errors)}")
    print(f"Logical clips         : {total_logical_clips}")
    print(f"HDF5 size this run    : {total_h5_gb:.3f} GiB")
    print(f"Free disk             : {free_space_gb(args.output_root):.2f} GiB")
    print(f"Output                : {args.output_root}")
    print(f"Summary               : {manifests_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
