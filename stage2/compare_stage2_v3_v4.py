from pathlib import Path
import json,csv
from config_stage2 import PROJECT_ROOT,OUTPUT_ROOT

V3=PROJECT_ROOT/"Stage_2_rPPG_FineTuning/results/cross_validation/cv_aggregate.json"
V4=OUTPUT_ROOT/"cross_validation/cv_aggregate.json"
OUT=OUTPUT_ROOT/"cross_validation/v3_vs_v4_morphology_stage2_metrics.csv"

def main():
    if not V3.is_file():raise FileNotFoundError(V3)
    if not V4.is_file():raise FileNotFoundError(V4)
    a=json.loads(V3.read_text());b=json.loads(V4.read_text())
    keys=[
      ("val_rmse_mean",False),
      ("val_pearson_r_mean",True),
      ("val_spectral_corr_mean",True),
      ("val_std_ratio_mean",None),
      ("val_best_lag_r_mean",True),
    ]
    rows=[]
    for k,up in keys:
        x=float(a[k]);y=float(b[k]);d=y-x
        if up is True:v="better" if d>0 else "worse"
        elif up is False:v="better" if d<0 else "worse"
        else:v="better" if abs(y-1)<abs(x-1) else "worse"
        rows.append({"metric":k,"stage2_v3":x,"stage2_v4_morph":y,"delta":d,"verdict":v})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print("="*100)
    print("Stage2 v3 vs v4 MorphologyPreserving — DEVELOPMENT CV ONLY")
    for r in rows:
        print(f"{r['metric']:28s} v3={r['stage2_v3']:.5f} v4={r['stage2_v4_morph']:.5f} delta={r['delta']:+.5f} {r['verdict']}")
    print("NOTE: final scientific decision requires re-running the SAME morphology audit on v4 OOF predictions.")
    print("Saved:",OUT)

if __name__=="__main__":main()
