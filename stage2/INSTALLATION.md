# Stage 2 v4: historical workstation setup

Source is the exact archived v4 code used on the original Linux workstation.
Do **not** use a MacBook for training or as proof of the historical environment.
Original source paths in `config_stage2.py` point to `/home/hajr/mcd_rppg/...`.
They are preserved verbatim to retain source-file hashes. For portability,
create your own working *copy* and update paths; do not label the changed copy
byte-identical to the archived source.

Training needs the original data, 10-fold Stage-1 checkpoints, the frozen 540/60
split, Data_QC_v2 Stage-2 ready manifest and checkpoints. Do not rerun a blind
test or regenerate the split. `run_stage2_v4_morph.sh` is the archived
*development-only* runner, and was written before the 118-epoch final fit.
It is not a complete end-to-end replication command.

For no-data smoke checks run `python -m unittest discover -s tests -v`
from repository root on a CPU/GPU environment with PyTorch installed.
