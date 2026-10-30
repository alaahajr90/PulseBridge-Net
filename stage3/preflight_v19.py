from pathlib import Path
import py_compile
from config_stage3 import *
from data_stage3 import raw_phys_morph_features_np
import numpy as np
here=Path(__file__).resolve().parent
files=['config_stage3.py','data_stage3.py','model_stage3.py','paired_loss_v19.py','train_stage3_cv.py','runtime_stage3.py','training_contract.py']
for f in files: py_compile.compile(str(here/f),doraise=True)
assert V11_VERSION=='Stage3_v19_CoupledD03'
assert V19_D03_INDEX==3 and V18_PHYS_FEATURE_NAMES[3]=='log_d1_rms'
x=np.sin(np.linspace(0,8*np.pi,128)); z=raw_phys_morph_features_np(x); assert len(z)==16 and np.isfinite(z).all()
for f in files:
    txt=(here/f).read_text()
    assert 'BLIND_CACHE_ROOT' not in txt or f=='config_stage3.py'
print('v19 PREFLIGHT PASS')
print('D03 =',V18_PHYS_FEATURE_NAMES[V19_D03_INDEX])
print('BLIND = LOCKED / NOT OPENED / NOT EVALUATED')
