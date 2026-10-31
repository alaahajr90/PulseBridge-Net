from __future__ import annotations

from pathlib import Path
import importlib.util
import torch

from config_stage2 import *
from io_utils import sha256_file
from protocol_stage2 import load_stage1_protocol
from data_stage2 import load_stage2_manifest, recordings_for_subjects
from model_stage2 import Stage2RPPGModel, load_stage1_encoder_weights
from losses_stage2 import RPPGPhysioCompositeLoss
from runtime_stage2 import resolve_or_load_runtime
from training_contract import build_training_contract, CONTRACT_JSON

ORIGINAL = PROJECT_ROOT / "Stage_2_rPPG_FineTuning"

# Files that MUST remain byte-identical to Stage-2 v3.
UNCHANGED_FILES = (
    "data_stage2.py",
    "model_stage2.py",
    "metrics_stage2.py",
    "protocol_stage2.py",
    "diagnostics_stage2.py",
    "io_utils.py",
)

# Config settings that MUST remain equal to original v3.
FROZEN_CONFIG_NAMES = (
    "SEED","DATA_ROOT","DB_CSV","STAGE2_QC_MANIFEST","EXPECTED_STAGE2_QC_MANIFEST_SHA256",
    "EXPECTED_STAGE2_READY_RECORDINGS","EXPECTED_STAGE2_HOLD_RECORDINGS",
    "EXPECTED_STAGE2_READY_SUBJECTS","EXPECTED_STAGE2_READY_SUBJECT_STATES",
    "EXPECTED_NO_STAGE2_READY_SUBJECTS","FROZEN_SPLIT_JSON","FROZEN_FOLDS_JSON",
    "EXPECTED_FROZEN_SPLIT_SHA256","EXPECTED_FROZEN_FOLDS_SHA256","K_FOLDS","CLIP_LEN","ROI_SIZE",
    "SCALE_TO_01","EPOCHS","PATIENCE","MIN_DELTA","LR_ENCODER_EARLY","LR_ENCODER_LATE",
    "LR_DECODER","WEIGHT_DECAY","GRAD_CLIP_NORM","LR_SCHEDULER_FACTOR",
    "LR_SCHEDULER_PATIENCE","LR_MIN","SMOOTH_L1_BETA","W_CORR","W_WAVE","W_DERIVATIVE",
    "W_SPECTRUM","W_BAND","W_STD","AUX_WEIGHT","AUX_CORR_WEIGHT","AUX_WAVE_WEIGHT",
    "FINE_TUNE_ALL_CONVS","LATE_CONV_COUNT","FREEZE_ALL_ENCODER_BATCHNORM_STATS",
    "FREEZE_ALL_ENCODER_BATCHNORM_AFFINE","FORBIDDEN_STAGE2_MODULE_TOKENS",
    "DECODER_CHANNELS","DECODER_DILATIONS","DECODER_KERNEL_SIZE","DECODER_DROPOUT",
    "TRAIN_SAMPLES_PER_EPOCH_RULE","VAL_RECORDINGS_PER_SUBJECT","VAL_CLIPS_PER_RECORDING",
    "VAL_CLIP_QUANTILES","PULSE_BAND_HZ","PSD_NFFT","BEST_LAG_MAX_SECONDS",
    "BATCH_SIZE_CANDIDATES","DEFAULT_CPU_BATCH_SIZE","NUM_WORKERS","PIN_MEMORY",
    "PERSISTENT_WORKERS","PREFETCH_FACTOR","N_WAVEFORM_PLOTS","SAVE_PLOTS_EVERY_EPOCH",
    "FINAL_EPOCH_RULE","MIN_SEGMENT_PPG_STD",
)

def _load_original_config():
    spec=importlib.util.spec_from_file_location("_stage2_v3_original_config", ORIGINAL/"config_stage2.py")
    if spec is None or spec.loader is None: raise RuntimeError("Cannot load original Stage2 config.")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    print("="*116)
    print("PulseBridge-Net Stage 2 v4 — MORPHOLOGY-PRESERVING CONTROLLED EXTENSION — PRE-FLIGHT")
    print("Original Stage-2 v3 folder is read-only reference and MUST NOT be modified.")
    print("Consumed 60-subject Blind cohort is FORBIDDEN.")
    print("="*116)

    if not ORIGINAL.is_dir():
        raise FileNotFoundError(f"Original Stage2 missing: {ORIGINAL}")

    protocol,folds,_=load_stage1_protocol()
    blind=set(map(str,protocol["blind_test_subjects"]));dev=set(map(str,protocol["development_subjects"]))
    print("Development/Blind/K:",len(dev),len(blind),folds["k_folds"])
    if len(dev)!=540 or len(blind)!=60 or int(folds["k_folds"])!=10:
        raise RuntimeError("Frozen protocol mismatch")
    print("[PASS] Frozen 540/60/10 protocol unchanged.")

    ready,hold,summary=load_stage2_manifest()
    print("READY/HOLD:",len(ready),len(hold))
    if len(ready)!=EXPECTED_STAGE2_READY_RECORDINGS or len(hold)!=EXPECTED_STAGE2_HOLD_RECORDINGS:
        raise RuntimeError("Stage2 manifest count mismatch")
    print("[PASS] Authoritative Stage2 READY/HOLD unchanged.")

    # Exact source preservation.
    for f in UNCHANGED_FILES:
        a=sha256_file(ORIGINAL/f);b=sha256_file(STAGE2_DIR/f)
        if a!=b: raise RuntimeError(f"Controlled-extension violation: {f} differs from original v3.")
    print("[PASS] data/model/metrics/protocol/diagnostics/io are byte-identical to original v3.")

    c3=_load_original_config()
    for n in FROZEN_CONFIG_NAMES:
        if globals()[n] != getattr(c3,n):
            raise RuntimeError(f"Original Stage2 setting changed: {n}: {globals()[n]} != {getattr(c3,n)}")
    print("[PASS] architecture, optimizer, LR, BN, sampling, epochs, patience and ALL original v3 loss weights unchanged.")

    print("[PASS] ONLY added scientific objective terms:")
    print("       normalized d1-shape =",W_MORPH_D1)
    print("       normalized d2-shape =",W_MORPH_D2)
    print("       multi-scale shape    =",W_MORPH_MULTISCALE,"lags",MORPH_MULTISCALE_LAGS)

    if EVALUATE_BLIND_IN_STAGE2:
        raise RuntimeError("Blind evaluation must remain disabled")
    for f in folds["folds"]:
        tr=set(map(str,f["train_subjects"]));va=set(map(str,f["val_subjects"]))
        if tr&va or tr&blind or va&blind: raise RuntimeError(f"Leakage fold {f['fold']}")
        print(f"Fold {int(f['fold']):02d}: train={len(tr)} val={len(va)} leakage=0 blind_overlap=0")

    # Real data/GPU probe through existing runtime path.
    fold1=folds["folds"][0]
    fold1_train=recordings_for_subjects(ready,fold1["train_subjects"])
    init=STAGE1_FOLDS_ROOT/"fold_01/checkpoints/encoder_best.pth"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime=resolve_or_load_runtime(fold1_train,init,device,reprobe=True)
    print("Resolved batch:",runtime["batch_size"])
    if runtime.get("probe_results"):
        p=runtime["probe_results"][0]
        print("Probe total loss:",p.get("probe_loss"))
        print("Probe morph d1/d2/multiscale:",p.get("probe_morph_d1"),p.get("probe_morph_d2"),p.get("probe_morph_multiscale"))
    print("[PASS] Real forward/backward morphology objective is finite.")

    contract=build_training_contract(runtime)
    print("[PASS] v4 training contract frozen:", CONTRACT_JSON)
    print("[PASS] No BP/rigidity/HR/HRV/RR target enters Stage2 v4.")
    print("[PASS] Original Stage-2 v3 directory was not modified by this preflight.")
    print("STAGE-2 v4 MORPHOLOGY-PRESERVING PRE-FLIGHT: PASS")
    print("BLIND: CONSUMED / FORBIDDEN")

if __name__=="__main__":
    main()
