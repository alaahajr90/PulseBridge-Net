import unittest
from pathlib import Path
import sys
import hashlib
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / 'stage2'
sys.path.insert(0, str(STAGE2))
if torch is not None:
    import config_stage2 as cfg
    from losses_stage2 import (
        RPPGPhysioCompositeLoss, _zshape, normalized_d1_shape_per_sample,
        normalized_d2_shape_per_sample, multiscale_shape_per_sample
    )

class TestStage2OriginalSource(unittest.TestCase):
    def test_original_source_hashes(self):
        expected = {
            'model_stage2.py': 'a86f2a144785ee44f6d81161e5f041109d7c1430860603592671a9d5e33bdd94',
            'losses_stage2.py': '74606bbbb98f30c9c0cf4ec8cde73a3699269962d7825ecbb6a3472fbd9cb27f',
            'config_stage2.py': 'f74ee2f733cf056cf8bbf1754cc2eaaf003881aa968d303043f2dfb05bbd8160',
            'train_stage2_cv.py': '8dfbc6ac0eb2584d758538251445b06e68e570ab46dab85a447076eabfa8360a',
        }
        for name, digest in expected.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((STAGE2 / name).read_bytes()).hexdigest(), digest)

@unittest.skipIf(torch is None, 'PyTorch unavailable')
class TestStage2OriginalLoss(unittest.TestCase):
    def test_weights_and_pixel_scale(self):
        self.assertEqual((cfg.W_MORPH_D1, cfg.W_MORPH_D2, cfg.W_MORPH_MULTISCALE), (.15, .35, .20))
        self.assertEqual(tuple(cfg.MORPH_MULTISCALE_LAGS), (1, 2, 4))
        self.assertFalse(cfg.SCALE_TO_01)

    def test_morphology_matches_independent_equations(self):
        torch.manual_seed(12)
        x = torch.randn(2, 128)
        y = torch.randn(2, 128)
        z = lambda a: (a - a.mean(dim=1,keepdim=True)) / (a.std(dim=1,keepdim=True,unbiased=False) + 1e-6)
        sl = lambda a,b: F.smooth_l1_loss(a,b,beta=1.,reduction='none').mean(1)
        d = lambda a: a[:,1:] - a[:,:-1]
        self.assertTrue(torch.allclose(normalized_d1_shape_per_sample(x,y), sl(z(d(x)),z(d(y)))))
        self.assertTrue(torch.allclose(normalized_d2_shape_per_sample(x,y), sl(z(d(d(x))),z(d(d(y))))))
        ms = torch.stack([sl(z(x[:,l:]-x[:,:-l]),z(y[:,l:]-y[:,:-l])) for l in (1,2,4)],dim=1).mean(1)
        self.assertTrue(torch.allclose(multiscale_shape_per_sample(x,y),ms))

    def test_full_loss_backward_and_exact_weights(self):
        torch.manual_seed(7)
        x = torch.randn(2,128,requires_grad=True)
        y = torch.randn(2,128)
        aux = [torch.randn(2,128,requires_grad=True) for _ in range(3)]
        loss, components = RPPGPhysioCompositeLoss()({'rppg':x,'aux_rppg':aux},y,torch.tensor([24.,30.]))
        expected = (3*components['corr_loss_per_sample'] + .75*components['wave_per_sample']
                   + .5*components['derivative_per_sample'] + .75*components['spectrum_per_sample']
                   + .05*components['band_per_sample'] + .25*components['std_loss_per_sample']
                   + .15*components['morph_d1_per_sample'] + .35*components['morph_d2_per_sample']
                   + .20*components['morph_multiscale_per_sample'])
        self.assertTrue(torch.allclose(expected,components['final_per_sample']))
        self.assertTrue(torch.allclose((expected+.15*components['aux_mean_per_sample']).mean(),loss))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(all(bool(torch.isfinite(a.grad).all()) for a in aux))

    def test_constant_waveform_stability(self):
        x=torch.ones(2,128)
        y=torch.zeros(2,128)
        for v in (normalized_d1_shape_per_sample(x,y),normalized_d2_shape_per_sample(x,y),multiscale_shape_per_sample(x,y)):
            self.assertTrue(bool(torch.isfinite(v).all()))

    def test_model_original_f7_interface(self):
        # Path override is applied only to this temporary unit-test process;
        # historical config_stage2.py on disk remains byte-identical.
        old=cfg.STAGE1_MODEL_PY
        try:
            cfg.STAGE1_MODEL_PY = ROOT / 'stage1' / 'model_stage1.py'
            from model_stage2 import Stage2RPPGModel
            model=Stage2RPPGModel().eval()
        finally:
            cfg.STAGE1_MODEL_PY=old
        torch.set_num_threads(2)
        with torch.no_grad():
            x=torch.randn(1,3,8,32,32)
            out=model(x)
            f7=model.forward_late_features(x)[-1]
        self.assertEqual(tuple(out['rppg'].shape),(1,8))
        self.assertEqual(tuple(f7.shape),(1,8,64))
        self.assertEqual(len(out['aux_rppg']),3)
        self.assertEqual(len(model._all_conv3d()),7)
        self.assertNotIn('f7',out)  # authentic source uses separate extraction API

if __name__ == '__main__':
    unittest.main()
