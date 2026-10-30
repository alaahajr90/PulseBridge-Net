#!/usr/bin/env python3
"""Print a privacy-conscious package and device inventory (no file scanning)."""
from __future__ import annotations
import importlib.metadata as md
import json
import platform
import subprocess
import sys

PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "h5py": "h5py",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "opencv-python": "opencv-python",
    "opencv-python-headless": "opencv-python-headless",
    "Pillow": "Pillow",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "pytest": "pytest",
    "dlib": "dlib",
}

def installed_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

report = {
    "label": "inspection_environment_not_proven_historical_training",
    "python": sys.version.split()[0],
    "os": platform.system(),
    "os_release": platform.release(),
    "architecture": platform.machine(),
    "packages": {key: installed_version(dist) for key, dist in PACKAGES.items()},
    "torch_cuda_build": None,
    "cuda_available": False,
    "gpu_names": [],
    "nvidia_smi_summary": None,
}
try:
    import torch
    report["torch_cuda_build"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if report["cuda_available"]:
        report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except (ImportError, RuntimeError, OSError) as exc:
    report["torch_inspection_error"] = type(exc).__name__
try:
    run = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if run.returncode == 0:
        report["nvidia_smi_summary"] = [line.strip() for line in run.stdout.splitlines() if line.strip()]
except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
    pass
print(json.dumps(report, indent=2, ensure_ascii=False))
