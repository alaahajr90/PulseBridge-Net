import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class QCv2SourceContract(unittest.TestCase):
    def test_original_extractor_sha256(self):
        proof = json.loads((ROOT / 'provenance/QC_V2_SOURCE.json').read_text())
        actual = hashlib.sha256((ROOT / 'preprocessing/extract_mcd_rppg_clips_compact.py').read_bytes()).hexdigest()
        self.assertEqual(actual, proof['original_extractor_sha256'])

    def test_source_module_present(self):
        q = ROOT / 'preprocessing/qcv2'
        for filename in ['common_qcv2.py', 'config_qcv2.py',
                         '02_full_audit_v2.py', '04_build_stage_specific_manifests.py',
                         '06_protocol_roster_audit.py', '07_final_preflight_v2.py']:
            self.assertTrue((q / filename).is_file(), filename)

    def test_no_individual_qc_results(self):
        q = ROOT / 'preprocessing/qcv2'
        self.assertFalse(list(q.rglob('*.csv')), 'Participant-level CSV must remain private')
        self.assertFalse(list(q.rglob('*.h5')), 'HDF5 recordings must remain private')

if __name__ == '__main__':
    unittest.main()
