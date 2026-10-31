"""Data-free SHA-256 contract for the 13 authenticated Stage-3 A2 files."""
from pathlib import Path
import hashlib
import json
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'provenance' / 'STAGE3_A2_ORIGINAL_SOURCE_SHA256.json'

class TestStage3A2OriginalSource(unittest.TestCase):
    def test_exact_historical_a2_source_hashes(self):
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        self.assertEqual(manifest['verified_file_count'], 13)
        self.assertEqual(len(manifest['files_sha256']), 13)
        for filename, expected in manifest['files_sha256'].items():
            with self.subTest(filename=filename):
                path = ROOT / 'stage3' / filename
                self.assertTrue(path.is_file(), filename)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_actual_a2_configuration_values(self):
        # Parse the historical original without importing it (absolute Linux paths).
        tree=ast.parse((ROOT/'stage3'/'config_stage3.py').read_text())
        values={}
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                try:values[node.targets[0].id]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
        self.assertEqual(values['SEED'],42)
        self.assertEqual(values['V11_VARIANCE_WEIGHT'],0.0)
        self.assertEqual(values['V13_PAIR_WEIGHT'],0.20)
        self.assertEqual(values['LR'],0.00015)
        self.assertEqual(values['WEIGHT_DECAY'],0.0002)
        self.assertEqual(values['EXPECTED_FROZEN_SPLIT_SHA256'],
            'd71d9c9e8ef75ef91f28cbac06b1b01de3325b95ecb3a992fb9ecb5afc7be739')
        self.assertEqual(values['EXPECTED_FROZEN_FOLDS_SHA256'],
            '628057141b979eb3af60595505a995b18d57e7dc4fd45e46d5d490ecb9f6c3e4')

    def test_provenance_qualifies_missing_final_trainer(self):
        manifest=json.loads(MANIFEST.read_text())
        self.assertFalse(manifest['final_trainer_source_located'])
        self.assertEqual(manifest['final_trainer_sha256_in_contract'],
            'f658fe2f8712606108322c709bbf2f04e0e0b01011b06910664de77437435616')

if __name__ == '__main__':
    unittest.main()
