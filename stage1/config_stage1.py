from pathlib import Path

# =============================================================================
# IMMUTABLE SCIENTIFIC PROTOCOL
# =============================================================================
SEED = 42

DB_CSV = Path("/home/hajr/mcd_rppg/db.csv")
DATA_ROOT = Path("/home/hajr/Desktop/mcd_rppg_clips")

STAGE1_ROOT = Path(
    "/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/"
    "Stage_1_PulseRepresentation"
)
OUTPUT_ROOT = STAGE1_ROOT / "results"

# The Stage-1 QC manifest produced BEFORE model training by Data_QC_v2.
STAGE1_QC_MANIFEST = Path(
    "/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/"
    "Data_QC_v2/results/04_stage1_manifest.csv"
)
EXPECTED_STAGE1_QC_MANIFEST_SHA256 = "70dfba7b07d6a9cc679d07b183d2b7cff17ee6d59033b65bec44dd61b8163878"

CLIP_LEN = 128
STRIDE = 8
ROI_SIZE = 64
SCALE_TO_01 = False

# Split protocol is SUBJECT-LEVEL and created from the original db.csv roster,
# never from the QC-filtered recording manifest.
BLIND_TEST_FRACTION = 0.10
K_FOLDS = 10

PROTOCOL_DIR = OUTPUT_ROOT / "protocol"
FROZEN_SPLIT_JSON = PROTOCOL_DIR / "frozen_subject_split.json"
FROZEN_FOLDS_JSON = PROTOCOL_DIR / "development_10folds.json"
PROTOCOL_MANIFEST_JSON = PROTOCOL_DIR / "protocol_manifest.json"

# =============================================================================
# TRAINING
# =============================================================================
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-5
GRAD_CLIP_NORM = 5.0

# The physical batch is resolved ONCE by preflight on the actual GPU, then frozen
# in results/preflight/resolved_runtime.json. 16 is tried first; smaller values
# are used only if a real forward+backward OOM occurs.
BATCH_SIZE_CANDIDATES = (16, 8, 4, 2)
DEFAULT_CPU_BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Balanced train epoch size: preserve approximately one pass over the available
# recording count, but allocate exposure equally across subjects.
TRAIN_SAMPLES_PER_EPOCH_RULE = "nearest_multiple_of_batch_to_n_recordings"

# Validation uses exactly one deterministic READY recording from BEFORE and one
# from AFTER per validation subject. Stage-1 QC guarantees both states for all
# 600 subjects. This equalizes validation contribution by subject and state.
VAL_RECORDINGS_PER_SUBJECT = 2

# =============================================================================
# SSL OBJECTIVE — preserved from the previous Stage-1 implementation
# =============================================================================
TEMPERATURE = 0.07
MARGIN = 0.30
TEMPORAL_WEIGHT = 0.25

PROJ_INPUT_DIM = 64
PROJ_HIDDEN_DIM = 64
PROJ_OUTPUT_DIM = 64
PROJ_DROPOUT = 0.10

HFLIP_PROB = 0.50
BRIGHTNESS_GAIN = 0.08
CONTRAST_GAIN = 0.08
GAUSSIAN_NOISE_STD = 1.0

CHECKPOINT_PRIMARY = "val_loss"
CHECKPOINT_SECONDARY = "val_cosine_gap"
MIN_ACCEPTABLE_EMBED_STD = 1e-3
FINAL_EPOCH_RULE = "median_best_epoch"

# Stage 1 remains strictly self-supervised. PPG is not loaded by the training
# dataset and is not used for loss, checkpointing, early stopping or diagnostics.
ENABLE_PULSE_PROBE = False

SAVE_PLOTS_EVERY_EPOCH = True

# Blind test is LOCKED throughout Stage 1. It is not evaluated here.
EVALUATE_BLIND_IN_STAGE1 = False

# Reproducibility / safety
REQUIRE_600_SUBJECTS = True
EXPECTED_TOTAL_SUBJECTS = 600
EXPECTED_STAGE1_READY_RECORDINGS = 3092
EXPECTED_STAGE1_SUBJECT_STATES = 1200
