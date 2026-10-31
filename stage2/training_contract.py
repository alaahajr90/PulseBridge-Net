from __future__ import annotations
from config_stage2 import *
from io_utils import save_json, sha256_file

CONTRACT_JSON = OUTPUT_ROOT / "protocol/stage2_training_contract.json"
ORIGINAL_V3_DIR = PROJECT_ROOT / "Stage_2_rPPG_FineTuning"

def stage1_encoder_paths():
    paths = {
        f"fold_{fold:02d}": STAGE1_FOLDS_ROOT / f"fold_{fold:02d}/checkpoints/encoder_best.pth"
        for fold in range(1, K_FOLDS + 1)
    }
    paths["final_development"] = STAGE1_FINAL_ENCODER
    return paths

def build_training_contract(runtime):
    code_files = [
        STAGE2_DIR/"config_stage2.py", STAGE2_DIR/"data_stage2.py",
        STAGE2_DIR/"model_stage2.py", STAGE2_DIR/"losses_stage2.py",
        STAGE2_DIR/"metrics_stage2.py", STAGE2_DIR/"protocol_stage2.py",
        STAGE2_DIR/"runtime_stage2.py", STAGE2_DIR/"train_stage2_cv.py",
    ]
    code_hashes = {p.name: sha256_file(p) for p in code_files}
    init_hashes = {name: sha256_file(path) for name, path in stage1_encoder_paths().items()}

    payload = {
        "contract_version": "PulseBridge_Stage2_v4_MorphologyPreserving",
        "controlled_experiment": True,
        "controlled_change": (
            "Original Stage-2 v3 architecture/data/protocol/optimizer/sampling and all original "
            "loss terms are preserved. Only normalized d1-shape, normalized d2-shape, and "
            "multi-scale temporal-increment morphology losses are added."
        ),
        "seed": SEED,
        "stage2_qc_manifest_sha256": sha256_file(STAGE2_QC_MANIFEST),
        "frozen_stage1_split_sha256": sha256_file(FROZEN_SPLIT_JSON),
        "frozen_stage1_folds_sha256": sha256_file(FROZEN_FOLDS_JSON),
        "stage1_model_py_sha256": sha256_file(STAGE1_MODEL_PY),
        "stage1_initialization_encoder_sha256": init_hashes,
        "physical_batch_size": int(runtime["batch_size"]),
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "lr_encoder_early": LR_ENCODER_EARLY,
        "lr_encoder_late": LR_ENCODER_LATE,
        "lr_decoder": LR_DECODER,
        "weight_decay": WEIGHT_DECAY,
        "original_v3_loss_weights": {
            "corr": W_CORR, "wave": W_WAVE, "derivative": W_DERIVATIVE,
            "spectrum": W_SPECTRUM, "band": W_BAND, "std": W_STD,
            "aux": AUX_WEIGHT,
        },
        "added_morphology_loss_weights": {
            "normalized_d1_shape": W_MORPH_D1,
            "normalized_d2_shape": W_MORPH_D2,
            "multiscale_shape": W_MORPH_MULTISCALE,
            "multiscale_lags": list(MORPH_MULTISCALE_LAGS),
        },
        "decoder": {
            "type": "attention-free residual dilated TCN (unchanged from v3)",
            "dilations": list(DECODER_DILATIONS),
            "kernel_size": DECODER_KERNEL_SIZE,
            "dropout": DECODER_DROPOUT,
        },
        "finetune_policy": "all Conv3d trainable; early/late discriminative LR; all encoder BN frozen",
        "training_sampling": "subject-balanced unique-subject batches",
        "validation_sampling": "deterministic up to BEFORE+AFTER per subject at 20/50/80 percent clips",
        "checkpoint_metric": "minimum validation TOTAL v4 composite loss; Pearson tie-break",
        "best_lag": "diagnostic only; never target realignment or training",
        "outputs": ["normalized_rppg_waveform"],
        "forbidden_outputs": ["blood_pressure","rigidity","heart_rate","hrv","respiratory_rate"],
        "attention_modules": False,
        "blind_policy": "CONSUMED_60_SUBJECT_BLIND_FORBIDDEN_FOR_v4",
        "morphology_audit_baseline": {
            "median_waveform_abs_r": 0.8392109097344008,
            "representative_beat_r": 0.9758646209757299,
            "d1_r": 0.9500241387241755,
            "d2_r": 0.6810333970879382,
            "median_morphology_feature_r": MORPHOLOGY_AUDIT_BASELINE_MEDIAN_R,
        },
        "code_sha256": code_hashes,
    }
    save_json(CONTRACT_JSON, payload)
    return payload
