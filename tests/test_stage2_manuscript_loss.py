import os
import sys
import unittest
from pathlib import Path
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "stage2"
sys.path.insert(0, str(STAGE2))
if torch is not None:
    from losses_stage2 import (RPPGPhysioCompositeLoss, z_normalize,
        first_difference, second_difference, morphology_terms_per_sample)
    from config_stage2 import (W_MORPH_FIRST, W_MORPH_SECOND,
        W_MORPH_MULTISCALE, MORPH_LAGS, SCALE_TO_01)

@unittest.skipIf(torch is None, "PyTorch unavailable; runtime test only")
class TestStage2PaperLoss(unittest.TestCase):
    def test_weights_and_pixel_scale(self):
        self.assertEqual((W_MORPH_FIRST, W_MORPH_SECOND, W_MORPH_MULTISCALE),
                         (0.15, 0.35, 0.20))
        self.assertEqual(tuple(MORPH_LAGS), (1, 2, 4))
        self.assertFalse(SCALE_TO_01)

    def test_differences_match_definitions(self):
        x=torch.tensor([[0., 1., 4., 9., 16., 25.]])
        self.assertTrue(torch.equal(first_difference(x), torch.tensor([[1.,3.,5.,7.,9.]])))
        self.assertTrue(torch.equal(second_difference(x), torch.tensor([[2.,2.,2.,2.]])))
        self.assertTrue(torch.equal(first_difference(x,2), torch.tensor([[4.,8.,12.,16.]])))

    def test_morphology_matches_independent_manual_eqs(self):
        torch.manual_seed(42)
        x=torch.randn(2,128)
        y=torch.randn(2,128)
        m1,m2,mms=morphology_terms_per_sample(x,y)
        z=lambda a: (a-a.mean(dim=1,keepdim=True))/(a.std(dim=1,keepdim=True,unbiased=False)+1e-6)
        sl=lambda a,b: F.smooth_l1_loss(a,b,beta=1.0,reduction="none").mean(dim=1)
        first=lambda a: a[:,1:]-a[:,:-1]
        second=lambda a: first(first(a))
        self.assertTrue(torch.allclose(m1,sl(z(first(x)),z(first(y)))))
        self.assertTrue(torch.allclose(m2,sl(z(second(x)),z(second(y)))))
        ms=torch.stack([sl(z(x[:,l:]-x[:,:-l]),z(y[:,l:]-y[:,:-l])) for l in (1,2,4)],1).mean(1)
        self.assertTrue(torch.allclose(mms,ms))

    def test_full_loss_and_backward(self):
        torch.manual_seed(11)
        prediction=torch.randn(2,128,requires_grad=True)
        reference=torch.randn(2,128)
        aux=[torch.randn(2,128,requires_grad=True) for _ in range(3)]
        loss,components=RPPGPhysioCompositeLoss()(
            {"rppg":prediction,"aux_rppg":aux}, reference, torch.tensor([24.,30.]))
        self.assertTrue(bool(torch.isfinite(loss)))
        expected=(3*components['corr_loss_per_sample']
                  +.75*components['wave_per_sample']
                  +.5*components['derivative_per_sample']
                  +.75*components['spectrum_per_sample']
                  +.05*components['band_per_sample']
                  +.25*components['std_loss_per_sample']
                  +.15*components['morph_first_per_sample']
                  +.35*components['morph_second_per_sample']
                  +.2*components['morph_multiscale_per_sample'])
        self.assertTrue(torch.allclose(expected,components['final_per_sample']))
        self.assertTrue(torch.allclose((expected+.15*components['aux_mean_per_sample']).mean(),loss))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        for a in aux: self.assertIsNotNone(a.grad)

    def test_constant_waveform_numerically_stable(self):
        x=torch.ones(2,128)
        y=torch.zeros(2,128)
        values=morphology_terms_per_sample(x,y)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in values))

    def test_stage2_model_f7(self):
        # Small synthetic clip to check shape; not a model-quality test.
        from model_stage2 import Stage2RPPGModel
        model=Stage2RPPGModel().eval()
        torch.set_num_threads(2)
        with torch.no_grad():
            out=model(torch.randn(1,3,8,32,32))
        self.assertEqual(tuple(out['rppg'].shape), (1,8))
        self.assertEqual(tuple(out['f7'].shape), (1,8,64))
        self.assertEqual(len(out['aux_rppg']),3)
        self.assertEqual(len(model._all_conv3d()),7)

if __name__ == '__main__':
    unittest.main()
