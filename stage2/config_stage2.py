from pathlib import Path

# =============================================================================
# IMMUTABLE SCIENTIFIC PROTOCOL
# =============================================================================
SEED = 42

PROJECT_ROOT = Path("/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net")
STAGE1_DIR = PROJECT_ROOT / "Stage_1_PulseRepresentation"
STAGE2_DIR = PROJECT_ROOT / "Stage_2_rPPG_FineTuning_v4_MorphologyPreserving"
OUTPUT_ROOT = STAGE2_DIR / "results"

DATA_ROOT = Path("/home/hajr/Desktop/mcd_rppg_clips")
DB_CSV = Path("/home/hajr/mcd_rppg/db.csv")

STAGE2_QC_MANIFEST = PROJECT_ROOT / "Data_QC_v2/results/04_stage2_manifest.csv"
EXPECTED_STAGE2_QC_MANIFEST_SHA256 = (
    "df13f0e60a67d2a79266b2af76e3bd2c3aa29a103c49877525a250368051d9b8"
)
EXPECTED_STAGE2_READY_RECORDINGS = 3079
EXPECTED_STAGE2_HOLD_RECORDINGS = 521
EXPECTED_STAGE2_READY_SUBJECTS = 599
EXPECTED_STAGE2_READY_SUBJECT_STATES = 1194
EXPECTED_NO_STAGE2_READY_SUBJECTS = ("4952",)

STAGE1_PROTOCOL_DIR = STAGE1_DIR / "results/protocol"
FROZEN_SPLIT_JSON = STAGE1_PROTOCOL_DIR / "frozen_subject_split.json"
FROZEN_FOLDS_JSON = STAGE1_PROTOCOL_DIR / "development_10folds.json"
PROTOCOL_MANIFEST_JSON = STAGE1_PROTOCOL_DIR / "protocol_manifest.json"

EXPECTED_FROZEN_SPLIT_SHA256 = (
    "d71d9c9e8ef75ef91f28cbac06b1b01de3325b95ecb3a992fb9ecb5afc7be739"
)
EXPECTED_FROZEN_FOLDS_SHA256 = (
    "628057141b979eb3af60595505a995b18d57e7dc4fd45e46d5d490ecb9f6c3e4"
)

STAGE1_FOLDS_ROOT = STAGE1_DIR / "results/folds"
STAGE1_FINAL_ENCODER = STAGE1_DIR / "results/final_model/stage1_encoder_for_stage2.pth"
STAGE1_BLIND_LOCK = STAGE1_DIR / "results/final_model/blind_test_LOCKED.json"
STAGE1_MODEL_PY = STAGE1_DIR / "model_stage1.py"

K_FOLDS = 10
CLIP_LEN = 128
ROI_SIZE = 64
SCALE_TO_01 = False

# =============================================================================
# STAGE-2 FINAL v3 — LEARNING CONFIGURATION
# =============================================================================
# v2 showed strong amplitude collapse / underfitting. v3 keeps the protocol
# frozen but upgrades only the waveform-learning machinery.
EPOCHS = 120
PATIENCE = 18
MIN_DELTA = 1e-4

# Discriminative fine-tuning: all Conv3d can adapt to physiology, but early
# layers move 5x slower than the final four Conv3d layers.
LR_ENCODER_EARLY = 1e-5
LR_ENCODER_LATE = 5e-5
LR_DECODER = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

# ReduceLROnPlateau acts only on development-fold validation composite loss.
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 6
LR_MIN = 1e-7

SMOOTH_L1_BETA = 1.0

# Physiological waveform objective:
# final = corr + waveform + derivative + spectrum + band + std
W_CORR = 3.0
W_WAVE = 0.75
W_DERIVATIVE = 0.50
W_SPECTRUM = 0.75
W_BAND = 0.05
W_STD = 0.25

# Auxiliary heads are intentionally weak in v3.
AUX_WEIGHT = 0.15
AUX_CORR_WEIGHT = 1.0
AUX_WAVE_WEIGHT = 0.25

# All Stage-1 Conv3d modules may adapt with discriminative LR.
FINE_TUNE_ALL_CONVS = True
LATE_CONV_COUNT = 4

# Keep BatchNorm frozen to avoid small-batch running-stat drift.
FREEZE_ALL_ENCODER_BATCHNORM_STATS = True
FREEZE_ALL_ENCODER_BATCHNORM_AFFINE = True

# Attention / HR / HRV / RR remain forbidden in Stage 2.
FORBIDDEN_STAGE2_MODULE_TOKENS = (
    "attn", "attention", "transformer", "heart_rate", "hrv",
    "respiratory", "respiration",
)

# Temporal decoder.
DECODER_CHANNELS = 64
DECODER_DILATIONS = (1, 2, 4, 8)
DECODER_KERNEL_SIZE = 5
DECODER_DROPOUT = 0.10

# =============================================================================
# SAMPLING / VALIDATION
# =============================================================================
TRAIN_SAMPLES_PER_EPOCH_RULE = "floor_to_batch_of_n_ready_recordings"
VAL_RECORDINGS_PER_SUBJECT = 2
VAL_CLIPS_PER_RECORDING = 3
VAL_CLIP_QUANTILES = (0.20, 0.50, 0.80)

# =============================================================================
# PHYSIOLOGICAL SIGNAL SETTINGS
# =============================================================================
PULSE_BAND_HZ = (0.5, 4.0)
PSD_NFFT = 512

# Diagnostic only; never used to align targets or train the model.
BEST_LAG_MAX_SECONDS = 1.0

# =============================================================================
# RUNTIME
# =============================================================================
BATCH_SIZE_CANDIDATES = (16, 8, 4, 2)
DEFAULT_CPU_BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

N_WAVEFORM_PLOTS = 4
SAVE_PLOTS_EVERY_EPOCH = True
FINAL_EPOCH_RULE = "median_best_epoch"
MIN_SEGMENT_PPG_STD = 1e-6

EVALUATE_BLIND_IN_STAGE2 = False


# =============================================================================
# STAGE-2 v4 MORPHOLOGY-PRESERVING CONTROLLED EXTENSION
# =============================================================================
# Baseline Stage-2 v3 waveform objective remains fully active and unchanged.
# These are the ONLY new scientific loss terms in v4.
#
# Rationale from the Development-only OOF morphology audit:
#   median |waveform r|       = 0.8392
#   representative-beat r    = 0.9759
#   first-derivative r        = 0.9500
#   second-derivative r       = 0.6810
#   morphology feature r      = 0.1688
#
# Therefore v4 adds shape-preserving supervision while keeping the architecture,
# dataset, split, optimizer, LR policy, BatchNorm policy and original v3 losses.
W_MORPH_D1 = 0.15
W_MORPH_D2 = 0.35
W_MORPH_MULTISCALE = 0.20
MORPH_MULTISCALE_LAGS = (1, 2, 4)
MORPH_NORM_EPS = 1e-6

# Development experiment only. The already-consumed 60-subject Blind cohort
# is forbidden for this v4 line of work.
EVALUATE_BLIND_IN_STAGE2 = False
MORPHOLOGY_AUDIT_BASELINE_MEDIAN_R = 0.16879113595892714
