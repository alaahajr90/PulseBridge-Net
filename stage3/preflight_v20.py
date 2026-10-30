from pathlib import Path
import torch
from config_stage3 import *
from model_stage3 import PulseBridgeStage3DualPathV7

def main():
    assert 'v20' in V11_VERSION.lower()
    assert V19_D03_INDEX==3 and V19_PAIR_INPUT_DIM==6
    m=PulseBridgeStage3DualPathV7()
    assert hasattr(m,'sbp_d03_correction_head')
    # zero-init contract inherited from v19
    last=[x for x in m.sbp_d03_correction_head.modules() if isinstance(x,torch.nn.Linear)][-1]
    assert torch.allclose(last.weight,torch.zeros_like(last.weight))
    assert torch.allclose(last.bias,torch.zeros_like(last.bias))
    txt=Path('paired_loss_v20.py').read_text()
    assert 'base_delta.detach()' in txt
    tr=Path('train_stage3_cv.py').read_text()
    assert "out_abs['sbp']=out['sbp_base_v19']" in tr
    assert 'paired_residual_d03_loss_v20' in tr
    print('v20 PREFLIGHT PASS')
    print('SBP pair target = true_delta - base_delta.detach()')
    print('Absolute SBP loss = UNCORRECTED base SBP')
    print('D03 = log_d1_rms')
    print('BLIND = LOCKED / NOT OPENED / NOT EVALUATED')
if __name__=='__main__': main()
