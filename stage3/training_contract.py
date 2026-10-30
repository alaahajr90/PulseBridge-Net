from pathlib import Path
from config_stage3 import *
from io_utils import sha256_file, save_json

CONTRACT_JSON = OUTPUT_ROOT / 'protocol/stage3_training_contract.json'


def build_contract(runtime):
    fold2 = {
        f'fold_{k:02d}': sha256_file(STAGE2_FOLDS_ROOT / f'fold_{k:02d}' / 'checkpoints' / STAGE2_FOLD_CHECKPOINT_NAME)
        for k in range(1, 11)
    }
    here = Path(__file__).resolve().parent
    tracked = (
        'config_stage3.py','data_stage3.py','model_stage3.py','model_stage3_v8_base.py',
        'losses_stage3.py','paired_loss_v19.py','paired_sampler_v11.py','train_stage3_cv.py',
        'metrics_stage3.py','runtime_stage3.py','training_contract.py'
    )
    code = {n: sha256_file(here / n) for n in tracked}
    payload = {
        'version': V11_VERSION,
        'seed': SEED,
        'stage3_manifest_sha256': sha256_file(STAGE3_QC_MANIFEST),
        'stage1_split_sha256': sha256_file(FROZEN_SPLIT_JSON),
        'stage1_folds_sha256': sha256_file(FROZEN_FOLDS_JSON),
        'stage2_fold_checkpoint_sha256': fold2,
        'stage2_final_sha256': None,
        'rppg_cache_root': str(CACHE_ROOT),
        'latent_cache_root': str(LATENT_CACHE_ROOT),
        'cache_policy': 'reuse exact v10 development rPPG + fold-matched Stage2-f7 latent caches; no blind cache',
        'unit_of_training': 'subject-state, paired Before/After batches',
        'unit_of_primary_evaluation': 'subject-state',
        'targets': ['SBP','DBP','dataset-native rigidity'],
        'architecture': 'v18 absolute backbone + zero-init pair-aware antisymmetric D03(log_d1_rms) SBP correction; DBP/rigidity v14 paired paths preserved',
        'v12_amp_ablation': {
            'raw_features_before_z_norm': list(AMP_FEATURE_NAMES),
            'feature_scaling': 'fit on TRAIN subjects only within each fold',
            'pooling': 'masked subject-state mean+SD across segments',
            'integration': 'zero-init target-specific residual calibration heads',
            'v11_waveform_path': 'unchanged z-normalized waveform path',
        },
        'absolute_loss': {
            'mae_weight': V11_MAE_WEIGHT,
            'rmse_weight': V11_RMSE_WEIGHT,
            'ccc_weight': V11_CCC_WEIGHT,
            'variance_log_std_weight': V11_VARIANCE_WEIGHT,
            'target_weights': [V11_SBP_WEIGHT,V11_DBP_WEIGHT,V11_STIFFNESS_WEIGHT],
        },
        'paired_loss': {
            'overall_weight': V14_PAIR_WEIGHT,
            'absolute_output_delta_huber_weight': V14_ABS_DELTA_WEIGHT,
            'aux_prepool_delta_huber_weight': V14_AUX_DELTA_WEIGHT,
            'delta_consistency_huber_weight': V14_CONSISTENCY_WEIGHT,
            'huber_beta': V14_DELTA_HUBER_BETA,
            'matching': 'exact (camera, clip_index) within subject Before/After',
            'bp_change_inputs': 'SBP: D03=log_d1_rms only, TRAIN-standardized, exact matched B/A/delta mean+SD (6D), antisymmetric correction to final SBP; DBP: v14 BP morphology + latent + amplitude descriptors',
            'rigidity_change_inputs': 'stiffness morphology segment feature + Stage2 latent segment embedding',
            'mini_batch_ccc_removed': True,
            'reason': 'batch=8 contains only four complete pairs; correlation gradient is unstable',
        },
        'checkpoint_selection': {'absolute_bp_weight': V19_SELECTION_ABS_WEIGHT, 'final_corrected_sbp_delta_weight': V19_SELECTION_FINAL_SBP_DELTA_WEIGHT},
        'cv_scope': CV_SCOPE,
        'blind_policy': 'LOCKED_NOT_OPENED_NOT_EVALUATED',
        'batch_size': runtime['batch_size'],
        'epochs_max': EPOCHS,
        'patience': PATIENCE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'code_sha256': code,
    }
    save_json(CONTRACT_JSON, payload)
    return payload
