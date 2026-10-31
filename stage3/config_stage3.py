from pathlib import Path
import os

SEED = 42
PROJECT_ROOT = Path("/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net")
STAGE1_DIR = PROJECT_ROOT / "Stage_1_PulseRepresentation"
STAGE2_DIR = PROJECT_ROOT / "Stage_2_rPPG_FineTuning_v4_MorphologyPreserving"
STAGE3_DIR = PROJECT_ROOT / "Stage_3_SupervisorModel_v20_FrozenResidualD03"
OUTPUT_ROOT = Path(os.environ["STAGE3_OUTPUT_ROOT_OVERRIDE"]) if os.environ.get("STAGE3_OUTPUT_ROOT_OVERRIDE") else STAGE3_DIR / "results"

# Reuse the exact frozen development caches used by v10.
CACHE_ROOT = PROJECT_ROOT / "Stage_3_SupervisorModel_v5_Stage2Morph/rppg_cache_stage2_v4_oldscale"
LATENT_CACHE_ROOT = PROJECT_ROOT / "Stage_3_SupervisorModel_v10_PairedLatentMorph/latent_cache_stage2_v4_f7_oldscale"
BLIND_CACHE_ROOT = STAGE3_DIR / "BLIND_DISABLED_DO_NOT_USE"

STAGE1_PROTOCOL_DIR = STAGE1_DIR / "results/protocol"
FROZEN_SPLIT_JSON = STAGE1_PROTOCOL_DIR / "frozen_subject_split.json"
FROZEN_FOLDS_JSON = STAGE1_PROTOCOL_DIR / "development_10folds.json"
PROTOCOL_MANIFEST_JSON = STAGE1_PROTOCOL_DIR / "protocol_manifest.json"
EXPECTED_FROZEN_SPLIT_SHA256 = "d71d9c9e8ef75ef91f28cbac06b1b01de3325b95ecb3a992fb9ecb5afc7be739"
EXPECTED_FROZEN_FOLDS_SHA256 = "628057141b979eb3af60595505a995b18d57e7dc4fd45e46d5d490ecb9f6c3e4"

STAGE3_QC_MANIFEST = PROJECT_ROOT / "Data_QC_v2/results/04_stage3_manifest.csv"
EXPECTED_STAGE3_QC_MANIFEST_SHA256 = "9ab7d652dd75c02bc8e682d3f608fadd1f6fb9844f8b369e16443e8564aefa53"
EXPECTED_STAGE3_TOTAL_RECORDINGS = 3600
EXPECTED_STAGE3_READY_RECORDINGS = 3092
EXPECTED_STAGE3_HOLD_RECORDINGS = 508
EXPECTED_STAGE3_READY_SUBJECTS = 600
EXPECTED_STAGE3_READY_SUBJECT_STATES = 1200
K_FOLDS = 10

STAGE2_FOLDS_ROOT = STAGE2_DIR / "results/folds"
STAGE2_FINAL_MODEL = STAGE2_DIR / "results/final_model/stage2_rppg_for_stage3.pt"
STAGE2_MODEL_PY = STAGE2_DIR / "model_stage2.py"
STAGE2_CONFIG_PY = STAGE2_DIR / "config_stage2.py"
STAGE2_FOLD_CHECKPOINT_NAME = "stage2_rppg_best.pth"

CLIP_LEN = 128
SEGMENTS_PER_RECORDING = 16
CACHE_INFERENCE_BATCH = 16
SBP_LABEL = "systolic_bp"
DBP_LABEL = "diastolic_bp"
STIFFNESS_LABEL = "rigidity"
CAMERAS = ("FullHDwebcam", "IriunWebcam", "USBVideo")
CAMERA_TO_INDEX = {c:i for i,c in enumerate(CAMERAS)}
CAMERA_EMBED_DIM = 12

# Keep the v10 architecture unchanged for a controlled v10 -> v11 comparison.
BASE_CH = 48
FEATURE_DIM = 128
FUSION_HIDDEN = 128
HEAD_HIDDEN = 96
DROPOUT = 0.15
QUALITY_TEMPERATURE = 0.35
QUALITY_MIN_LOGIT = -5.0
QUALITY_MAX_LOGIT = 5.0

LATENT_DIM = 64
LATENT_TEMPORAL_LEN = 128
LATENT_HIDDEN = 64

BATCH_SIZE_CANDIDATES = (8,4,2,1)
DEFAULT_CPU_BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

EPOCHS = int(os.environ.get("STAGE3_EPOCHS_OVERRIDE","180"))
PATIENCE = int(os.environ.get("STAGE3_PATIENCE_OVERRIDE","28"))
MIN_DELTA = 1e-5
LR = 1.5e-4
WEIGHT_DECAY = 2e-4
GRAD_CLIP_NORM = 5.0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 7
LR_MIN = 1e-6

# v11 active absolute-target objective.
V11_MAE_WEIGHT = 0.55
V11_RMSE_WEIGHT = 0.30
V11_CCC_WEIGHT = 0.15
V11_VARIANCE_WEIGHT = 0.0
V11_VARIANCE_EPS = 1e-6

# Target weights retained exactly from v10.
V11_SBP_WEIGHT = 0.40
V11_DBP_WEIGHT = 0.35
V11_STIFFNESS_WEIGHT = 0.25

# v11 paired Before->After objective.
PAIR_DELTA_WEIGHT = 0.15
PAIR_DELTA_MAE_WEIGHT = 0.75
PAIR_DELTA_CCC_WEIGHT = 0.25
PAIR_DELTA_EPS = 1e-6
PAIR_DELTA_MIN_PAIRS_FOR_CCC = 3

# Model-selection objective: retain dominant absolute BP criterion while
# requiring the saved checkpoint to respect Before->After response magnitude.
V11_SELECTION_ABS_WEIGHT = 0.80
V11_SELECTION_DELTA_WEIGHT = 0.20

BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05
EVALUATE_BLIND_DURING_TRAINING = False
CV_SCOPE = "downstream_conditional_cv_using_fold_matched_frozen_stage2_models"
V11_VERSION = "Stage3_v20_FrozenResidualD03"

# v12 controlled amplitude/scale-preservation ablation.
# ONLY new information relative to v11: four pre-z-normalization scale descriptors
# per cached rPPG segment, standardized using TRAIN subjects only.
AMP_FEATURE_NAMES = ("log_std", "log_rms", "log_peak_to_peak", "log_d1_rms")
AMP_FEATURE_DIM = 4
AMP_SUMMARY_DIM = AMP_FEATURE_DIM * 2  # masked mean + masked SD across segments
AMP_HIDDEN = 32
AMP_EPS = 1e-8

# v13 representation-level paired supervision.
# Motivation: v12 widened SBP range but did not improve subject-specific delta correlation.
# Tiny-batch delta CCC is intentionally removed from the training objective because batch=8
# contains only four complete subject pairs, making correlation gradients noisy.
V13_PAIR_WEIGHT = 0.20
V13_ABS_DELTA_WEIGHT = 0.45
V13_AUX_DELTA_WEIGHT = 0.40
V13_CONSISTENCY_WEIGHT = 0.15
V13_DELTA_HUBER_BETA = 0.50
V13_PAIR_EMBED_HIDDEN = 64

# v14 controlled pre-pooling paired-change experiment.
# Absolute v12/v13 path is preserved; only the auxiliary change representation is replaced.
V14_PAIR_WEIGHT = 0.20
V14_ABS_DELTA_WEIGHT = 0.45
V14_AUX_DELTA_WEIGHT = 0.40
V14_CONSISTENCY_WEIGHT = 0.15
V14_DELTA_HUBER_BETA = 0.50
V14_CHANGE_HIDDEN = 96
V14_MIN_MATCHED_SEGMENTS = 1

# v16 controlled intervention: direct PRE-Z state-pair SBP change branch.
V16_VERSION = "v16_DirectPreZSBP"
V16_SBP_CHANGE_INPUT = "state_amp_summary_before_after_delta"
V16_SBP_CHANGE_HIDDEN = 64
V16_SBP_DIRECT_ONLY = True
V16_DBP_CHANGE_PATH = "v14_unchanged"
V16_STIFFNESS_CHANGE_PATH = "v14_unchanged"

# v17: paired scale-aware morphology intervention for SBP.
V17_VERSION = "v17_PairedScaleAwareMorph"
V17_PREZ_FEATURE_DIM = 128
V17_PAIR_HIDDEN = 96
V17_SBP_AUX_HIDDEN = 64
# Pair loss formulation: final absolute-output delta + new scale-aware auxiliary + consistency.
V17_SBP_ABS_DELTA_WEIGHT = 0.45
V17_SBP_AUX_DELTA_WEIGHT = 0.40
V17_SBP_CONSISTENCY_WEIGHT = 0.15
# Checkpoint selection uses the new auxiliary SBP delta directly, while retaining absolute BP.
V17_SELECTION_ABS_WEIGHT = 0.80
V17_SELECTION_AUX_SBP_WEIGHT = 0.20


# v18: fixed physiologically constrained PRE-Z morphology descriptors for paired SBP.
V18_VERSION = "v18_PhysMorph"
V18_PHYS_FEATURE_NAMES = (
    "log_std","log_rms","log_ptp","log_d1_rms",
    "skew","kurtosis","d1_abs_mean","d2_abs_mean",
    "q10","q25","q75","q90",
    "iqr","upper_lower_asym","ac_peak","ac_peak_lag_norm"
)
V18_PHYS_FEATURE_DIM = len(V18_PHYS_FEATURE_NAMES)
# pair vector = B mean+sd, A mean+sd, matched delta mean+sd = 6*D
V18_PAIR_INPUT_DIM = 6 * V18_PHYS_FEATURE_DIM
V18_SBP_HIDDEN = 64
V18_SBP_ABS_DELTA_WEIGHT = 0.45
V18_SBP_AUX_DELTA_WEIGHT = 0.40
V18_SBP_CONSISTENCY_WEIGHT = 0.15
V18_SELECTION_ABS_WEIGHT = 0.80
V18_SELECTION_AUX_SBP_WEIGHT = 0.20


# v19: controlled pair-coupled D03 correction for deployable paired SBP.
V19_VERSION = "v19_CoupledD03"
V19_D03_INDEX = 3  # log_d1_rms in V18_PHYS_FEATURE_NAMES
V19_PAIR_INPUT_DIM = 6  # B mean/sd, A mean/sd, matched delta mean/sd for D03
V19_CORR_HIDDEN = 16
V19_SBP_FINAL_DELTA_WEIGHT = 1.0
V19_SELECTION_ABS_WEIGHT = 0.80
V19_SELECTION_FINAL_SBP_DELTA_WEIGHT = 0.20

# v20: residual-target D03 correction. Pair loss cannot backpropagate into the absolute SBP backbone.
V20_VERSION = "v20_FrozenResidualD03"
V20_SBP_RESIDUAL_WEIGHT = 1.0
V20_SELECTION_ABS_WEIGHT = 0.80
V20_SELECTION_FINAL_SBP_DELTA_WEIGHT = 0.20
