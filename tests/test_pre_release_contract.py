from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestPreReleaseContract(unittest.TestCase):
    def test_all_python_parses(self):
        files = list(ROOT.rglob('*.py'))
        self.assertGreaterEqual(len(files), 30)
        for f in files:
            with self.subTest(f=f.relative_to(ROOT)):
                ast.parse(f.read_text(encoding='utf-8-sig'), filename=str(f))

    def test_selected_stage3_a2(self):
        s=(ROOT/'stage3'/'config_stage3.py').read_text()
        self.assertIn('V11_VARIANCE_WEIGHT = 0.0',s)
        self.assertIn('EXPECTED_FROZEN_SPLIT_SHA256',s)
        self.assertIn('EXPECTED_FROZEN_FOLDS_SHA256',s)

    def test_stage1_source_present(self):
        self.assertTrue((ROOT/'stage1'/'model_stage1.py').exists())
        self.assertTrue((ROOT/'stage1'/'train_stage1_cv.py').exists())

    def test_stage2_and_original_qc_sources_present(self):
        self.assertTrue((ROOT/'stage2'/'model_stage2.py').exists())
        self.assertTrue((ROOT/'preprocessing'/'extract_mcd_rppg_clips_compact.py').is_file())
        self.assertTrue((ROOT/'preprocessing'/'qcv2'/'07_final_preflight_v2.py').is_file())
        self.assertTrue((ROOT/'docs'/'QC_V2_REPORT.md').is_file())
        self.assertIn('PRE-RELEASE', (ROOT/'README.md').read_text())

if __name__ == '__main__':
    unittest.main()
