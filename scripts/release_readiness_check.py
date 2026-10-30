#!/usr/bin/env python3
"""Check necessary release inputs without opening any participant data."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
checks = {
    "stage1 code": (ROOT / "stage1" / "model_stage1.py").is_file(),
    "stage2 code": (ROOT / "stage2" / "model_stage2.py").is_file(),
    "stage3 code": (ROOT / "stage3" / "model_stage3.py").is_file(),
    "canonical Data_QC_v2 source": any(p.is_file() and p.suffix == ".py" for p in (ROOT / "preprocessing").rglob("*.py")),
    "license chosen (LICENSE file)": any((ROOT / n).is_file() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt")),
}
for title, ok in checks.items():
    print(("PASS" if ok else "BLOCKED") + "  " + title)
print("Manual scientific gates: historical Stage-2 v4, A2 run contracts, frozen splits, end-to-end integration, rights review.")
if not all(checks.values()):
    print("RESULT: NOT READY for an unconditional public reproduction claim")
    sys.exit(1)
print("RESULT: BASIC FILE GATES PASS; manual scientific review still required")
