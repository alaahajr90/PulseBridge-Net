from pathlib import Path
import py_compile, numpy as np
from config_stage3 import *
from data_stage3 import raw_phys_morph_features_np
here=Path(__file__).resolve().parent
checks=[]
def ck(n,x): checks.append((n,bool(x))); print(('PASS' if x else 'FAIL'),'-',n)
ck('isolated v18 directory',here.name=='Stage_3_SupervisorModel_v18_PhysMorph')
ck('config points to v18',STAGE3_DIR.resolve()==here.resolve())
ck('blind disabled',EVALUATE_BLIND_DURING_TRAINING is False)
ck('blind cache disabled path','BLIND_DISABLED_DO_NOT_USE' in str(BLIND_CACHE_ROOT))
ck('frozen dev cache',CACHE_ROOT.exists())
ck('frozen latent cache',LATENT_CACHE_ROOT.exists())
d=(here/'data_stage3.py').read_text();m=(here/'model_stage3.py').read_text();l=(here/'paired_loss_v18.py').read_text();t=(here/'train_stage3_cv.py').read_text()
ck('fixed 16D morphology extractor','raw_phys_morph_features_np' in d and 'V18_PHYS_FEATURE_DIM' in d)
ck('TRAIN-only morphology scaler','class PhysMorphFeatureScaler' in d and "fi['train_subjects']" in t)
ck('no learned v17 PRE-Z waveform encoder','self.sbp_prez_encoder' not in m)
ck('no SBP attention pair pool','sbp_prez_change_pool' not in m)
ck('exact camera+clip matching retained','_matched_indices' in m and "zip(meta_b['cameras'],meta_b['clip_indices'])" in m)
ck('fixed SBP pair summary','ms(ph_b)' in m and 'ms(ph_a-ph_b)' in m)
ck('SBP final+aux+consistency objective','V18_SBP_ABS_DELTA_WEIGHT*abs_t[0]' in l and 'V18_SBP_AUX_DELTA_WEIGHT*aux_t[0]' in l and 'V18_SBP_CONSISTENCY_WEIGHT*con_t[0]' in l)
ck('DBP/RIG v14 retained','V14_ABS_DELTA_WEIGHT*abs_t[1]' in l and 'V14_ABS_DELTA_WEIGHT*abs_t[2]' in l)
ck('selection sees v18 auxiliary','val_sbp_aux_delta_mae' in t and 'V18_SELECTION_AUX_SBP_WEIGHT' in t)
f=raw_phys_morph_features_np(np.sin(np.linspace(0,8*np.pi,128))+.15*np.sin(np.linspace(0,16*np.pi,128)))
ck('descriptor smoke test finite 16D',f.shape==(16,) and np.isfinite(f).all())
for fn in ['config_stage3.py','data_stage3.py','model_stage3.py','paired_loss_v18.py','train_stage3_cv.py','runtime_stage3.py','training_contract.py']:
    try: py_compile.compile(str(here/fn),doraise=True);ok=True
    except Exception as e: print(e);ok=False
    ck('syntax '+fn,ok)
if not all(x for _,x in checks): raise SystemExit('PREFLIGHT FAILED')
print('\nPREFLIGHT PASS — v18 PhysMorph')
print('BLIND = LOCKED / NOT OPENED / NOT EVALUATED')
