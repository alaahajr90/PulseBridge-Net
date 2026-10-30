# Dependency map and example exact versions

The versions below are a **suggested Python 3.10 Linux installation profile**.
They have **not** been verified against the original training logs, and the
complete original environment has not been supplied. `torch==2.5.1` is an
example platform-specific installation described in `INSTALLATION.md`.

| Package/import | Suggested exact version | Repository purpose |
|---|---|---|
| Python | 3.10.x | interpreter used on the available Linux audit environment |
| `torch` (PyTorch) | 2.5.1 | Conv3D, Conv1D, losses, SSL and vascular regression; CPU/CUDA wheel chosen separately |
| `numpy` | 1.26.4 | clip data, arrays, descriptors, vectorized computations |
| `scipy` | 1.13.1 | signal processing / statistics dependency for the broader pipeline |
| `h5py` | 3.11.0 | HDF5 ROI and PPG cache input |
| `pandas` | 2.2.2 | manifests, metadata tables, metrics |
| `scikit-learn` | 1.5.2 | downstream metrics / scaler-related utilities |
| `opencv-python-headless` (`cv2`) | 4.10.0.84 | OpenCV support, mainly for video/QC workflows; no GUI |
| `Pillow` (`PIL`) | 10.4.0 | figure and image handling |
| `matplotlib` | 3.9.2 | diagnostics and study figures |
| `tqdm` | 4.66.5 | training/processing progress |
| `pytest` | 8.3.3 | optional extra unit-test runner |

The built-in `unittest` runner itself requires **no external test package**.
The repository's current Stage-1, Stage-2 and Stage-3 training code imports
`torch`, `numpy`, `h5py`, `pandas`, `matplotlib`, `PIL`, `tqdm`, and
`sklearn` (Stage 3). `scipy` and OpenCV are retained for the broader processing
and original project dependency contract. `dlib` is **not pinned** because the
original canonical `Data_QC_v2` implementation was not provided; if it is
indeed imported, document and version it when that source is restored.

**Important distinctions:**

1. These pins are **illustrative**: `requirements-recommended.txt` is not an
   archival environment lock; pip transitive dependency resolution can change.
2. PyTorch/CUDA wheels depend on the operating system, architecture and NVIDIA
   driver. The CUDA toolkit system version should not be conflated with
   `torch.version.cuda`.
3. To reproduce the original experiment environment, collect the historical
   `pip freeze`, `conda list --explicit`, GPU model and NVIDIA driver version,
   plus package hashes where possible, and audit them against dated run logs.
4. Do not infer training reproducibility from an import test or 10 passing
   synthetic tests.
