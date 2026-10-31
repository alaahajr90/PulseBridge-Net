"""Data_QC_v2 configuration, portable adaptation of the supplied source.

Original ``config_qcv2.py`` SHA-256 is recorded in provenance/QC_V2_SOURCE.json.
This file changes only filesystem configuration and not QC thresholds.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("PULSEBRIDGE_DATA_ROOT", "/PATH/TO/MCD-rPPG"))
CLIP_ROOT = Path(os.environ.get("PULSEBRIDGE_CLIP_ROOT", "/PATH/TO/mcd_rppg_clips"))
EXTRACTOR_PATH = Path(os.environ.get(
    "PULSEBRIDGE_EXTRACTOR", str(PROJECT_ROOT / "preprocessing" / "extract_mcd_rppg_clips_compact.py")
))
PREDICTOR_PATH = Path(os.environ.get("PULSEBRIDGE_LANDMARKS", "/PATH/TO/shape_predictor_68_face_landmarks.dat"))
DB_CSV = Path(os.environ.get("PULSEBRIDGE_DB_CSV", str(DATA_ROOT / "db.csv")))
ERRORS_CSV = Path(os.environ.get("PULSEBRIDGE_ERRORS_CSV", str(CLIP_ROOT / "_manifests" / "errors.csv")))

RESULTS = PACKAGE_ROOT / "results"
LOGS = PACKAGE_ROOT / "logs"
BACKUPS = PACKAGE_ROOT / "backups"
TEMP_ROOT = PACKAGE_ROOT / "_repair_tmp"

EXPECTED_SUBJECTS = 600
EXPECTED_RECORDINGS = 3600
EXPECTED_SUBJECT_STATES = 1200
CLIP_LEN = 128
STRIDE = 8
ROI_SIZE = 64
GZIP_LEVEL = 4
MIN_FREE_GB = 30.0
ROI_FALLBACK_MAX = 0.10
SYNC_GOOD_MEAN_MAX_S = 0.010
SYNC_GOOD_MAX_MAX_S = 0.100
SYNC_REVIEW_MEAN_MAX_S = 0.020
SYNC_REVIEW_MAX_MAX_S = 2.000
STAGE3_LABELS = ("systolic_bp", "diastolic_bp", "rigidity")
STAGE1_PROTOCOL = Path(os.environ.get("PULSEBRIDGE_STAGE1_PROTOCOL", str(PROJECT_ROOT / "stage1" / "results" / "protocol")))
