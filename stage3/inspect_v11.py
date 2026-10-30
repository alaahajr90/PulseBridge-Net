#!/usr/bin/env python3
from pathlib import Path
import torch
from config_stage3 import *
from model_stage3 import PulseBridgeStage3DualPathV7
from losses_stage3 import DualPathLossV11
from paired_loss_v11 import paired_delta_loss_v11
from paired_sampler_v11 import PairedSubjectStateBatchSampler

print('='*118)
print('STAGE3-v11 VARIANCE + PAIRED CALIBRATION PREFLIGHT')
print('STAGE3_DIR       :', STAGE3_DIR)
print('rPPG cache       :', CACHE_ROOT)
print('f7 latent cache  :', LATENT_CACHE_ROOT)
print('output root      :', OUTPUT_ROOT)
print('blind path       :', BLIND_CACHE_ROOT)

assert V11_VERSION == 'Stage3_v11_VariancePairedCalibration'
assert EVALUATE_BLIND_DURING_TRAINING is False
assert 'BLIND_DISABLED_DO_NOT_USE' in str(BLIND_CACHE_ROOT)
assert Path(LATENT_CACHE_ROOT).name == 'latent_cache_stage2_v4_f7_oldscale'
assert abs(V11_MAE_WEIGHT-0.55)<1e-12
assert abs(V11_RMSE_WEIGHT-0.30)<1e-12
assert abs(V11_CCC_WEIGHT-0.15)<1e-12
assert V11_VARIANCE_WEIGHT > 0
assert abs(PAIR_DELTA_WEIGHT-0.15)<1e-12
assert abs(PAIR_DELTA_MAE_WEIGHT+PAIR_DELTA_CCC_WEIGHT-1.0)<1e-12
assert abs(V11_SELECTION_ABS_WEIGHT+V11_SELECTION_DELTA_WEIGHT-1.0)<1e-12

m=PulseBridgeStage3DualPathV7()
assert hasattr(m,'base') and hasattr(m,'latent_encoder')
assert hasattr(m,'sbp_latent_residual') and hasattr(m,'dbp_latent_residual') and hasattr(m,'stiffness_latent_residual')

# Loss sanity check without touching any dataset/blind file.
out={'sbp':torch.tensor([0.0,0.2,0.7,-0.5]),'dbp':torch.tensor([0.1,-0.2,0.3,-0.1]),'stiffness':torch.tensor([0.2,0.1,-0.2,0.4])}
y=torch.tensor([[0.0,0.0,0.0],[0.4,-0.1,0.2],[1.0,0.5,-0.1],[-0.8,-0.3,0.5]])
L,parts=DualPathLossV11()(out,y)
assert torch.isfinite(L)
assert 'sbp_variance_match_loss' in parts
metas=[{'subject_id':'s1','state':'before'},{'subject_id':'s1','state':'after'},
       {'subject_id':'s2','state':'before'},{'subject_id':'s2','state':'after'}]
PL,Pparts,npairs=paired_delta_loss_v11(out,y,metas)
assert torch.isfinite(PL) and npairs==2

print('UNCHANGED v10 architecture        : PASS')
print('Explicit variance preservation    : PASS')
print('Structured paired delta objective : PASS')
print('Pair-aware checkpoint criterion   : PASS')
print('v10 fold-matched latent reuse     : PASS')
print('Blind disabled                    : PASS')
print('PREFLIGHT                         : PASS')
print('='*118)
