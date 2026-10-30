# Installation / replacement

Because this is a protocol-level revision, replace the old Stage-2 source folder
instead of copying individual `.py` files into an old result tree.

Recommended:

```bash
cd /home/hajr/mcd_rppg/code_HR_RR_model/PulseBridge_Net

mv Stage_2_rPPG_FineTuning Stage_2_rPPG_FineTuning_PRE_FINAL_v2_BACKUP

unzip Stage_2_rPPG_FineTuning_FINAL_v2.zip
```

The ZIP contains the folder:

`Stage_2_rPPG_FineTuning/`

Do not copy old Stage-2 `results/` into the new folder. Stage 1 and Data_QC_v2
remain untouched and are read through absolute paths.
