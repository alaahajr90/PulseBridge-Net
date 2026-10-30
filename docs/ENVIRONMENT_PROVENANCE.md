# Environment provenance and hardware reporting

## Known and unknown facts

The available independent Linux code check used an environment named
`icbme2026_mamba` and ran **Python 3.10**. That observation identifies a
**current inspection environment** only. It is not proof of the versions
installed when the 10 CV folds, the 118-epoch Stage-2 final fit or the
13-epoch A2 Stage-3 final fit were originally performed.

Do **not** record unverified package versions as historical training facts.

## Verification command

On the Linux research workstation, activate the **actual experiment**
environment and run:

```bash
python scripts/capture_environment.py
python -m pip freeze > pip-freeze-unreviewed.txt
conda list --explicit > conda-explicit-unreviewed.txt
nvidia-smi
```

The Python script emits the major package versions, GPU name(s), PyTorch CUDA
runtime, driver details when `nvidia-smi` is available, and availability flags.
It deliberately does not collect participant files, home directory paths,
or secret environment variables.

Before public release, verify the environment against dated training contracts
and check the exported text for sensitive paths, internal package indexes,
private VCS repository URLs, credentials, or other identifying metadata. The
historical environment should then be archived with a provenance date and
linked to the corresponding run and checkpoint hashes.

## Suggested paper/repository metadata

Once confirmed, report:

- OS distribution and version; Python; package manager; exact package versions.
- GPU model and VRAM (if GPU execution); NVIDIA driver; PyTorch build and
  `torch.version.cuda` or CPU-only indication.
- CUDA/cuDNN availability (only when measured), random seeds, deterministic
  flags, batch sizes and data-loader settings.
- Stage-specific training contracts; exact frozen split, manifest and
  checkpoint SHA-256 hashes; cache provenance.
- Explicit indication of which tests were synthetic checks versus actual
  development-fold evaluation; no repeated independent test.
