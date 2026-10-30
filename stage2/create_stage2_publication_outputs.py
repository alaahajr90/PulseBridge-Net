from pathlib import Path
import json
import statistics
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/home/hajr/mcd_rppg/code_HR_RR_model/"
    "PulseBridge_Net/Stage_2_rPPG_FineTuning"
)

RESULTS = ROOT / "results"

CV_SUMMARY = RESULTS / "cross_validation" / "cv_summary.csv"
CV_AGGREGATE = RESULTS / "cross_validation" / "cv_aggregate.json"
FINAL_SUMMARY = RESULTS / "final_model" / "final_summary.json"

FIG_DIR = ROOT / "figures" / "stage2"
TABLE_DIR = ROOT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONVERGENCE = FIG_DIR / "stage2_cv_convergence.pdf"
OUT_QUALITY = FIG_DIR / "stage2_waveform_quality.pdf"
OUT_STATE = FIG_DIR / "stage2_state_comparison.pdf"
OUT_CAMERA = FIG_DIR / "stage2_camera_comparison.pdf"
OUT_WAVEFORMS = FIG_DIR / "stage2_representative_waveforms.pdf"
OUT_MANIFEST = FIG_DIR / "stage2_figure_manifest.json"

OUT_MAIN_TABLE = TABLE_DIR / "stage2_cv_results.tex"
OUT_GROUP_TABLE = TABLE_DIR / "stage2_group_results.tex"


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# HELPERS
# ============================================================

def mean_sd(values):
    values = [float(v) for v in values]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt_mean_sd(mean, sd, decimals=4):
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def robust_representative_fold(df):
    """
    Select representative fold based on robust normalized distance
    to cross-fold medians.

    This is intentionally not the best-performing fold.
    """

    metrics = [
        "val_total_loss",
        "val_rmse",
        "val_pearson_r",
        "val_spectral_corr",
        "val_std_ratio",
        "best_epoch",
    ]

    tmp = df[metrics].copy()

    med = tmp.median()

    mad = (
        tmp.sub(med)
        .abs()
        .median()
        .replace(0, 1e-12)
    )

    distance = (
        tmp.sub(med)
        .abs()
        .div(mad)
        .mean(axis=1)
    )

    idx = distance.idxmin()

    return int(df.loc[idx, "fold"])


# ============================================================
# LOAD CV SUMMARY
# ============================================================

if not CV_SUMMARY.exists():
    raise FileNotFoundError(CV_SUMMARY)

if not CV_AGGREGATE.exists():
    raise FileNotFoundError(CV_AGGREGATE)

cv_summary = pd.read_csv(CV_SUMMARY)

with open(CV_AGGREGATE, "r", encoding="utf-8") as f:
    cv_aggregate = json.load(f)

with open(FINAL_SUMMARY, "r", encoding="utf-8") as f:
    final_summary = json.load(f)

folds = sorted(cv_summary["fold"].astype(int).tolist())

if len(folds) != 10:
    raise RuntimeError(
        f"Expected 10 folds, found {len(folds)}: {folds}"
    )

median_best_epoch = int(cv_aggregate["median_best_epoch"])


# ============================================================
# LOAD TRAINING HISTORIES
# ============================================================

histories = {}

required_columns = {
    "fold",
    "epoch",
    "train_total_loss",
    "val_total_loss",
    "val_mae",
    "val_rmse",
    "val_pearson_r",
    "val_abs_pearson_r",
    "val_cosine",
    "val_derivative_rmse",
    "val_spectral_corr",
    "val_pred_std",
    "val_target_std",
    "val_std_ratio",
    "val_best_lag_r",
    "val_best_lag_seconds",
}

for fold in folds:

    path = (
        RESULTS
        / "folds"
        / f"fold_{fold:02d}"
        / "training_history.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Fold {fold:02d} missing columns: {sorted(missing)}"
        )

    histories[fold] = (
        df.sort_values("epoch")
        .reset_index(drop=True)
    )


# ============================================================
# TRAINING LENGTHS
# ============================================================

training_lengths = {
    fold: int(df["epoch"].max())
    for fold, df in histories.items()
}

common_max_epoch = min(training_lengths.values())

print("\nTraining lengths:")

for fold in folds:
    print(
        f"  Fold {fold:02d}: "
        f"{training_lengths[fold]} epochs"
    )

print(
    f"\nCommon epoch range: 1--{common_max_epoch}"
)

print(
    f"Median best epoch: {median_best_epoch}"
)


# ============================================================
# BEST-CHECKPOINT METRICS
# ============================================================

best_rows = []

for _, row in cv_summary.iterrows():

    fold = int(row["fold"])
    best_epoch = int(row["best_epoch"])

    hist = histories[fold]

    match = hist[
        hist["epoch"].astype(int) == best_epoch
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Could not uniquely locate "
            f"Fold {fold:02d}, epoch {best_epoch}"
        )

    h = match.iloc[0]

    best_rows.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "val_total_loss": float(h["val_total_loss"]),
        "val_mae": float(h["val_mae"]),
        "val_rmse": float(h["val_rmse"]),
        "val_pearson_r": float(h["val_pearson_r"]),
        "val_abs_pearson_r": float(h["val_abs_pearson_r"]),
        "val_cosine": float(h["val_cosine"]),
        "val_derivative_rmse": float(h["val_derivative_rmse"]),
        "val_spectral_corr": float(h["val_spectral_corr"]),
        "val_pred_std": float(h["val_pred_std"]),
        "val_target_std": float(h["val_target_std"]),
        "val_std_ratio": float(h["val_std_ratio"]),
        "val_best_lag_r": float(h["val_best_lag_r"]),
        "val_best_lag_seconds": float(h["val_best_lag_seconds"]),
    })


best_df = (
    pd.DataFrame(best_rows)
    .sort_values("fold")
    .reset_index(drop=True)
)


# ============================================================
# AGGREGATE OVERALL METRICS
# ============================================================

overall_metrics = {}

metric_map = {
    "Validation total loss": "val_total_loss",
    "MAE": "val_mae",
    "RMSE": "val_rmse",
    "Pearson r": "val_pearson_r",
    "Absolute Pearson r": "val_abs_pearson_r",
    "Cosine similarity": "val_cosine",
    "Derivative RMSE": "val_derivative_rmse",
    "Spectral correlation": "val_spectral_corr",
    "Predicted SD": "val_pred_std",
    "Target SD": "val_target_std",
    "SD ratio": "val_std_ratio",
    "Best-lag correlation": "val_best_lag_r",
    "Best-lag offset (s)": "val_best_lag_seconds",
}

for label, column in metric_map.items():
    m, s = mean_sd(best_df[column])
    overall_metrics[label] = {
        "mean": m,
        "sd": s
    }


best_epoch_min = int(best_df["best_epoch"].min())
best_epoch_max = int(best_df["best_epoch"].max())


# ============================================================
# GROUP DIAGNOSTICS
# ============================================================

group_data = {
    "camera": {},
    "state": {},
}

group_metrics = [
    "total_loss",
    "mae",
    "rmse",
    "pearson_r",
    "abs_pearson_r",
    "cosine",
    "derivative_rmse",
    "spectral_corr",
    "std_ratio",
    "best_lag_r",
    "best_lag_seconds",
]

for fold in folds:

    path = (
        RESULTS
        / "folds"
        / f"fold_{fold:02d}"
        / "best_group_diagnostics.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)

    for domain in ["camera", "state"]:

        for group_name, metrics in g[domain].items():

            if group_name not in group_data[domain]:
                group_data[domain][group_name] = {
                    metric: []
                    for metric in group_metrics
                }

            for metric in group_metrics:
                group_data[domain][group_name][metric].append(
                    float(metrics[metric])
                )


# ============================================================
# GROUP AGGREGATES
# ============================================================

group_summary = {
    "camera": {},
    "state": {},
}

for domain in ["camera", "state"]:

    for group_name, metrics in group_data[domain].items():

        group_summary[domain][group_name] = {}

        for metric, values in metrics.items():

            m, s = mean_sd(values)

            group_summary[domain][group_name][metric] = {
                "mean": m,
                "sd": s
            }


# ============================================================
# REPRESENTATIVE FOLD
# ============================================================

representative_fold = robust_representative_fold(
    best_df
)

print(
    f"\nRepresentative fold: "
    f"Fold {representative_fold:02d}"
)


# ============================================================
# FIGURE 1
# CROSS-VALIDATION CONVERGENCE
# ============================================================

epochs = np.arange(
    1,
    common_max_epoch + 1
)

train_matrix = []
val_matrix = []

for fold in folds:

    df = histories[fold]

    common = (
        df[df["epoch"] <= common_max_epoch]
        .sort_values("epoch")
    )

    train_matrix.append(
        common["train_total_loss"].to_numpy()
    )

    val_matrix.append(
        common["val_total_loss"].to_numpy()
    )


train_matrix = np.vstack(train_matrix)
val_matrix = np.vstack(val_matrix)

train_mean = train_matrix.mean(axis=0)
train_sd = train_matrix.std(axis=0, ddof=1)

val_mean = val_matrix.mean(axis=0)
val_sd = val_matrix.std(axis=0, ddof=1)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.2),
    constrained_layout=True
)


ax = axes[0]

ax.plot(
    epochs,
    train_mean,
    linewidth=1.8,
    label="Training loss"
)

ax.fill_between(
    epochs,
    train_mean - train_sd,
    train_mean + train_sd,
    alpha=0.16
)

ax.plot(
    epochs,
    val_mean,
    linewidth=1.9,
    label="Validation loss"
)

ax.fill_between(
    epochs,
    val_mean - val_sd,
    val_mean + val_sd,
    alpha=0.16
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Composite waveform loss")
ax.set_title("(a) Cross-validation convergence")

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="upper right"
)


ax = axes[1]

x = best_df["fold"].to_numpy()
y = best_df["best_epoch"].to_numpy()

ax.plot(
    x,
    y,
    marker="o",
    linewidth=1.2
)

ax.axhline(
    median_best_epoch,
    linestyle="--",
    linewidth=1.4,
    label=f"Median = {median_best_epoch}"
)

ax.set_xticks(x)

ax.set_xlabel("Cross-validation fold")
ax.set_ylabel("Selected best epoch")

ax.set_title(
    "(b) Fold-specific checkpoint epochs"
)

ax.grid(
    axis="y",
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="lower left"
)


fig.savefig(
    OUT_CONVERGENCE,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2
# WAVEFORM QUALITY
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.6, 4.0),
    constrained_layout=True
)

x = best_df["fold"].to_numpy()


# ------------------------------------------------------------
# Pearson r
# ------------------------------------------------------------

ax = axes[0]

vals = best_df["val_pearson_r"].to_numpy()
m = vals.mean()
s = vals.std(ddof=1)

ax.plot(
    x,
    vals,
    marker="o",
    linewidth=1.2
)

ax.axhline(
    m,
    linestyle="--",
    linewidth=1.3,
    label=f"Mean = {m:.3f}"
)

ax.set_xticks(x)
ax.set_xlabel("Fold")
ax.set_ylabel("Pearson r")
ax.set_title("(a) Temporal waveform agreement")

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False
)

ax.text(
    0.03,
    0.05,
    f"Mean ± SD = {m:.3f} ± {s:.3f}",
    transform=ax.transAxes,
    fontsize=8.5
)


# ------------------------------------------------------------
# Spectral correlation
# ------------------------------------------------------------

ax = axes[1]

vals = best_df["val_spectral_corr"].to_numpy()
m = vals.mean()
s = vals.std(ddof=1)

ax.plot(
    x,
    vals,
    marker="o",
    linewidth=1.2
)

ax.axhline(
    m,
    linestyle="--",
    linewidth=1.3,
    label=f"Mean = {m:.3f}"
)

ax.set_xticks(x)
ax.set_xlabel("Fold")
ax.set_ylabel("Spectral correlation")
ax.set_title("(b) Pulse-band spectral agreement")

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False
)

ax.text(
    0.03,
    0.05,
    f"Mean ± SD = {m:.3f} ± {s:.3f}",
    transform=ax.transAxes,
    fontsize=8.5
)


# ------------------------------------------------------------
# SD ratio
# ------------------------------------------------------------

ax = axes[2]

vals = best_df["val_std_ratio"].to_numpy()
m = vals.mean()
s = vals.std(ddof=1)

ax.plot(
    x,
    vals,
    marker="o",
    linewidth=1.2
)

ax.axhline(
    1.0,
    linestyle=":",
    linewidth=1.1,
    label="Target ratio = 1"
)

ax.axhline(
    m,
    linestyle="--",
    linewidth=1.3,
    label=f"Mean = {m:.3f}"
)

ax.set_xticks(x)
ax.set_xlabel("Fold")
ax.set_ylabel("Predicted/target SD ratio")
ax.set_title("(c) Signal-variability preservation")

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False
)

ax.text(
    0.03,
    0.05,
    f"Mean ± SD = {m:.3f} ± {s:.3f}",
    transform=ax.transAxes,
    fontsize=8.5
)


fig.savefig(
    OUT_QUALITY,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3
# BEFORE / AFTER STATE COMPARISON
# ============================================================

state_order = [
    "before",
    "after",
]

metrics_for_state = [
    ("pearson_r", "Pearson r"),
    ("rmse", "RMSE"),
    ("spectral_corr", "Spectral correlation"),
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.8, 3.8),
    constrained_layout=True
)

for ax, (metric, ylabel) in zip(
    axes,
    metrics_for_state
):

    means = [
        group_summary["state"][state][metric]["mean"]
        for state in state_order
    ]

    sds = [
        group_summary["state"][state][metric]["sd"]
        for state in state_order
    ]

    ax.bar(
        ["Before", "After"],
        means,
        yerr=sds,
        capsize=4
    )

    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.6
    )


fig.savefig(
    OUT_STATE,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 4
# CAMERA COMPARISON
# ============================================================

camera_order = [
    "FullHDwebcam",
    "IriunWebcam",
    "USBVideo",
]

camera_labels = [
    "FullHD",
    "Iriun",
    "USB",
]

metrics_for_camera = [
    ("pearson_r", "Pearson r"),
    ("rmse", "RMSE"),
    ("spectral_corr", "Spectral correlation"),
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.4, 3.9),
    constrained_layout=True
)

for ax, (metric, ylabel) in zip(
    axes,
    metrics_for_camera
):

    means = [
        group_summary["camera"][cam][metric]["mean"]
        for cam in camera_order
    ]

    sds = [
        group_summary["camera"][cam][metric]["sd"]
        for cam in camera_order
    ]

    ax.bar(
        camera_labels,
        means,
        yerr=sds,
        capsize=4
    )

    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)

    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.6
    )


fig.savefig(
    OUT_CAMERA,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 5
# REPRESENTATIVE WAVEFORMS
# ============================================================

source_waveform_plot = (
    RESULTS
    / "folds"
    / f"fold_{representative_fold:02d}"
    / "plots"
    / "best_waveforms.png"
)

if not source_waveform_plot.exists():
    raise FileNotFoundError(
        source_waveform_plot
    )

img = Image.open(
    source_waveform_plot
).convert("RGB")

fig, ax = plt.subplots(
    1,
    1,
    figsize=(10.8, 6.2),
    constrained_layout=True
)

ax.imshow(img)
ax.axis("off")

ax.set_title(
    f"Representative Stage-2 waveform reconstructions "
    f"(Fold {representative_fold:02d})"
)

fig.savefig(
    OUT_WAVEFORMS,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# LATEX TABLE 1
# MAIN STAGE-2 CV RESULTS
# ============================================================

main_table = rf"""
\begin{{table}}[t]
\centering
\caption{{
Stage~2 subject-wise 10-fold cross-validation results.
Values are reported as mean $\pm$ standard deviation across folds.
Waveform error metrics are computed on the segment-wise standardized
PPG scale.
}}
\label{{tab:stage2_cv_results}}

\small
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{7pt}}

\begin{{tabular}}{{lc}}
\toprule
\textbf{{Metric}} & \textbf{{Mean $\pm$ SD}} \\
\midrule

Validation composite loss
&
{fmt_mean_sd(
    overall_metrics["Validation total loss"]["mean"],
    overall_metrics["Validation total loss"]["sd"],
    4
)}
\\

MAE
&
{fmt_mean_sd(
    overall_metrics["MAE"]["mean"],
    overall_metrics["MAE"]["sd"],
    4
)}
\\

RMSE
&
{fmt_mean_sd(
    overall_metrics["RMSE"]["mean"],
    overall_metrics["RMSE"]["sd"],
    4
)}
\\

Pearson $r$
&
{fmt_mean_sd(
    overall_metrics["Pearson r"]["mean"],
    overall_metrics["Pearson r"]["sd"],
    4
)}
\\

Absolute Pearson $r$
&
{fmt_mean_sd(
    overall_metrics["Absolute Pearson r"]["mean"],
    overall_metrics["Absolute Pearson r"]["sd"],
    4
)}
\\

Cosine similarity
&
{fmt_mean_sd(
    overall_metrics["Cosine similarity"]["mean"],
    overall_metrics["Cosine similarity"]["sd"],
    4
)}
\\

Derivative RMSE
&
{fmt_mean_sd(
    overall_metrics["Derivative RMSE"]["mean"],
    overall_metrics["Derivative RMSE"]["sd"],
    4
)}
\\

Spectral correlation
&
{fmt_mean_sd(
    overall_metrics["Spectral correlation"]["mean"],
    overall_metrics["Spectral correlation"]["sd"],
    4
)}
\\

Predicted/target SD ratio
&
{fmt_mean_sd(
    overall_metrics["SD ratio"]["mean"],
    overall_metrics["SD ratio"]["sd"],
    4
)}
\\

Best-lag correlation$^\dagger$
&
{fmt_mean_sd(
    overall_metrics["Best-lag correlation"]["mean"],
    overall_metrics["Best-lag correlation"]["sd"],
    4
)}
\\

Best-lag offset (s)$^\dagger$
&
{fmt_mean_sd(
    overall_metrics["Best-lag offset (s)"]["mean"],
    overall_metrics["Best-lag offset (s)"]["sd"],
    4
)}
\\

\midrule

Median selected epoch
&
{median_best_epoch}
\\

Selected-epoch range
&
{best_epoch_min}--{best_epoch_max}
\\

\bottomrule
\end{{tabular}}

\vspace{{1mm}}
\begin{{minipage}}{{0.93\linewidth}}
\footnotesize
$^\dagger$Best-lag metrics are diagnostic only and were not used for
target realignment, training, or checkpoint selection.
\end{{minipage}}

\end{{table}}
""".strip()


OUT_MAIN_TABLE.write_text(
    main_table,
    encoding="utf-8"
)


# ============================================================
# LATEX TABLE 2
# STATE + CAMERA SUBGROUP RESULTS
# ============================================================

def subgroup_row(name, domain, key):

    g = group_summary[domain][key]

    return (
        f"{name} & "
        f"{g['mae']['mean']:.4f} $\\pm$ {g['mae']['sd']:.4f} & "
        f"{g['rmse']['mean']:.4f} $\\pm$ {g['rmse']['sd']:.4f} & "
        f"{g['pearson_r']['mean']:.4f} $\\pm$ {g['pearson_r']['sd']:.4f} & "
        f"{g['spectral_corr']['mean']:.4f} $\\pm$ {g['spectral_corr']['sd']:.4f} & "
        f"{g['std_ratio']['mean']:.4f} $\\pm$ {g['std_ratio']['sd']:.4f} \\\\"
    )


group_table = r"""
\begin{table*}[t]
\centering
\caption{
Stage~2 reconstruction performance stratified by physiological state
and acquisition camera. Values are mean $\pm$ standard deviation across
the 10 subject-wise development folds.
}
\label{tab:stage2_group_results}

\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{5pt}

\begin{tabular}{lccccc}
\toprule
\textbf{Group}
&
\textbf{MAE}
&
\textbf{RMSE}
&
\textbf{Pearson $r$}
&
\textbf{Spectral corr.}
&
\textbf{SD ratio}
\\
\midrule

\multicolumn{6}{l}{\textit{Physiological state}} \\

""" + "\n".join([
    subgroup_row(
        "Before exercise",
        "state",
        "before"
    ),
    subgroup_row(
        "After exercise",
        "state",
        "after"
    ),
]) + r"""

\midrule

\multicolumn{6}{l}{\textit{Acquisition camera}} \\

""" + "\n".join([
    subgroup_row(
        "FullHD webcam",
        "camera",
        "FullHDwebcam"
    ),
    subgroup_row(
        "Iriun webcam",
        "camera",
        "IriunWebcam"
    ),
    subgroup_row(
        "USB video",
        "camera",
        "USBVideo"
    ),
]) + r"""

\bottomrule
\end{tabular}
\end{table*}
""".strip()


OUT_GROUP_TABLE.write_text(
    group_table,
    encoding="utf-8"
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "source_results": str(RESULTS),

    "n_folds": len(folds),

    "common_epoch_range": [
        1,
        common_max_epoch
    ],

    "median_best_epoch": median_best_epoch,

    "best_epoch_range": [
        best_epoch_min,
        best_epoch_max
    ],

    "representative_fold": representative_fold,

    "overall_metrics": overall_metrics,

    "group_summary": group_summary,

    "final_model": final_summary,

    "outputs": {
        "cv_convergence": str(OUT_CONVERGENCE),
        "waveform_quality": str(OUT_QUALITY),
        "state_comparison": str(OUT_STATE),
        "camera_comparison": str(OUT_CAMERA),
        "representative_waveforms": str(OUT_WAVEFORMS),
        "main_table": str(OUT_MAIN_TABLE),
        "group_table": str(OUT_GROUP_TABLE),
    }
}


with open(
    OUT_MANIFEST,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STAGE-2 PUBLICATION OUTPUTS CREATED")
print("=" * 80)

print("\nOverall CV metrics:")

for label in [
    "Validation total loss",
    "MAE",
    "RMSE",
    "Pearson r",
    "Absolute Pearson r",
    "Cosine similarity",
    "Derivative RMSE",
    "Spectral correlation",
    "SD ratio",
    "Best-lag correlation",
    "Best-lag offset (s)",
]:

    d = overall_metrics[label]

    print(
        f"  {label:28s}: "
        f"{d['mean']:.6f} ± {d['sd']:.6f}"
    )


print("\nPhysiological state:")

for state in ["before", "after"]:

    d = group_summary["state"][state]

    print(
        f"  {state:8s}: "
        f"r={d['pearson_r']['mean']:.4f} ± "
        f"{d['pearson_r']['sd']:.4f}, "
        f"RMSE={d['rmse']['mean']:.4f} ± "
        f"{d['rmse']['sd']:.4f}, "
        f"Spectral={d['spectral_corr']['mean']:.4f} ± "
        f"{d['spectral_corr']['sd']:.4f}"
    )


print("\nCamera:")

for cam in camera_order:

    d = group_summary["camera"][cam]

    print(
        f"  {cam:14s}: "
        f"r={d['pearson_r']['mean']:.4f} ± "
        f"{d['pearson_r']['sd']:.4f}, "
        f"RMSE={d['rmse']['mean']:.4f} ± "
        f"{d['rmse']['sd']:.4f}, "
        f"Spectral={d['spectral_corr']['mean']:.4f} ± "
        f"{d['spectral_corr']['sd']:.4f}"
    )


print(
    f"\nMedian best epoch: "
    f"{median_best_epoch}"
)

print(
    f"Best-epoch range: "
    f"{best_epoch_min}--{best_epoch_max}"
)

print(
    f"Representative fold: "
    f"fold_{representative_fold:02d}"
)

print("\nGenerated figures:")

for p in [
    OUT_CONVERGENCE,
    OUT_QUALITY,
    OUT_STATE,
    OUT_CAMERA,
    OUT_WAVEFORMS,
]:

    print(f"  {p}")


print("\nGenerated tables:")

print(f"  {OUT_MAIN_TABLE}")
print(f"  {OUT_GROUP_TABLE}")

print("\nManifest:")
print(f"  {OUT_MANIFEST}")

print("=" * 80)
