from pathlib import Path
import py_compile
from config_stage3 import *
here=Path(__file__).resolve().parent
checks=[]
def ck(name,ok):
    checks.append((name,bool(ok))); print(("PASS" if ok else "FAIL"),"-",name)

ck("isolated v16 directory", here.name=="Stage_3_SupervisorModel_v16_DirectPreZSBP")
ck("config points to v16", STAGE3_DIR.resolve()==here.resolve())
ck("blind disabled", EVALUATE_BLIND_DURING_TRAINING is False)
ck("frozen development cache exists", CACHE_ROOT.exists())
ck("frozen latent cache exists", LATENT_CACHE_ROOT.exists())
m=(here/"model_stage3.py").read_text()
l=(here/"paired_loss_v16.py").read_text()
t=(here/"train_stage3_cv.py").read_text()
ck("direct PRE-Z SBP head present","sbp_direct_prez_head" in m)
ck("Before+After+Delta 24-D contract","AMP_SUMMARY_DIM*3" in m and "sa-sb" in m)
ck("SBP direct-only pair objective","sbp = aux_t[0]" in l)
ck("SBP not consistency-forced","sbp = aux_t[0]" in l and "V14_CONSISTENCY_WEIGHT*con_t[0]" not in l)
ck("DBP v14 objective retained","V14_ABS_DELTA_WEIGHT*abs_t[1]" in l and "V14_AUX_DELTA_WEIGHT*aux_t[1]" in l)
ck("Rigidity v14 objective retained","V14_ABS_DELTA_WEIGHT*abs_t[2]" in l and "V14_AUX_DELTA_WEIGHT*aux_t[2]" in l)
ck("pair total weight unchanged",abs(V14_PAIR_WEIGHT-0.20)<1e-12)
ck("selection weights unchanged",abs(V11_SELECTION_ABS_WEIGHT-.80)<1e-12 and abs(V11_SELECTION_DELTA_WEIGHT-.20)<1e-12)
ck("training imports v16 loss","paired_loss_v16" in t)
for f in ["config_stage3.py","model_stage3.py","paired_loss_v16.py","train_stage3_cv.py","training_contract.py"]:
    try: py_compile.compile(str(here/f),doraise=True); ok=True
    except Exception as e: print(e); ok=False
    ck("syntax "+f,ok)
if not all(v for _,v in checks): raise SystemExit("PREFLIGHT FAILED")
print("\nPREFLIGHT PASS — v16 DirectPreZSBP controlled Fold1 experiment")
print("BLIND = LOCKED / NOT OPENED / NOT EVALUATED")
