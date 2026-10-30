# PulseBridge-Net: installation and smoke checks

> **Status:** manuscript-aligned source candidate, **not** a validated
> reproduction of the manuscript results. These commands cover source checks
> and environment creation. They do not recreate missing data/QC artifacts,
> final checkpoints, or the fixed evaluation protocol.

## 1. Intended system

- Linux x86_64 for the original research pipeline; the development workstation
  previously used an Anaconda/Miniconda environment called `icbme2026_mamba`
  and Python **3.10**. The exact original `torch`, CUDA and other installed
  versions have not been recovered from an environment export.
- NVIDIA GPU is recommended for video training; CPU is sufficient for static
  checks and many small synthetic tests. No minimum GPU memory claim has been
  validated for full development training.
- Disk requirements depend on the **separately obtained** MCD-rPPG dataset,
  precomputed ROI/PPG HDF5 clips and Stage-2 rPPG/F7 caches. The dataset is
  **not** redistributed with this repository.

## 2. Create a new isolated example environment

Using Conda (do not modify the historical research environment):

```bash
conda create -n pulsebridge-example python=3.10 pip -y
conda activate pulsebridge-example
python -m pip install --upgrade pip
```

### Install PyTorch separately

The following commands are **example wheels** for PyTorch 2.5.1 on Linux.
Choose **one** route suited to the local NVIDIA driver. Do not run them all.
The official listing also offers CUDA 11.8 and CUDA 12.4 variants:

**CUDA 12.1 example:**

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only example:**

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

See [official PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
for the correct wheel selection. The installed NVIDIA driver and GPU must
support the selected runtime. `nvidia-smi` reports driver capability; it
**does not itself identify which CUDA runtime PyTorch was installed with**.

### Install the remaining suggested exact versions

```bash
python -m pip install -r requirements-recommended.txt
python -m pip check
```

`requirements-recommended.txt` contains proposed compatible versions, **not**
an exported package set from the original 2026 experiments. The repository's
original `requirements.txt` remains as a less-constrained list of dependencies.
The recommendation uses `opencv-python-headless` for Linux servers. If an
OpenCV GUI is required, replace it with a matching `opencv-python` release;
never install both variants together. Neither variant repairs the missing
`Data_QC_v2` implementation. A historical dlib face detector may require a
separate system-specific installation; it is not assumed to be part of the
canonical QC pipeline without the original source.

## 3. Verify the environment and current source

```bash
python --version
python -c "import torch; print('torch:',torch.__version__,'torch CUDA:',torch.version.cuda,'GPU:',torch.cuda.is_available())"
python -m unittest discover -s tests -v
python scripts/capture_environment.py
python scripts/release_readiness_check.py
```

The unit tests are **synthetic/static tests only**. They cannot establish the
published SBP, DBP or arterial-stiffness performance. They do not exercise an
end-to-end restored MCD-rPPG pipeline. A failure in `release_readiness_check`
while the original QC is missing is **expected and intentional**.

## 4. Paths and data contracts

Historical `stage1/config_stage1.py`, `stage2/config_stage2.py` and
`stage3/config_stage3.py` refer to paths such as:

```text
/home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net/
/home/hajr/Desktop/mcd_rppg_clips/
```

Those historical absolute paths are **not portable**. Before real execution,
restore the original subject-independent split and exact QC manifest/hash
contracts. Do not regenerate splits from QC-filtered recordings or create a
new 60-subject test set. Stage 3 depends on fold-matched Stage-2 waveform and
F7 caches; the manuscript's one-time independent test is not an ordinary
repeatable validation script.

## 5. Capture the true historical environment (research workstation)

Activate the environment that actually ran the experiments, then run:

```bash
conda activate icbme2026_mamba
python scripts/capture_environment.py
python -m pip freeze > environment-original-pip-freeze.txt
conda list --explicit > environment-original-conda-explicit.txt
nvidia-smi
```

**Do not commit the exports without checking them first**: package references,
absolute paths, internal mirrors or editable VCS dependencies can expose
private locations or credentials. Review, sanitize, date and attach the records
as a provenance artifact only after verifying that this was the historical
training environment. It must not be represented as such merely because it
is active today.
