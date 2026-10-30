from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/home/hajr/mcd_rppg/code_HR_RR_model/"
    "PulseBridge_Net/Stage_1_PulseRepresentation"
)

RESULTS = ROOT / "results"

CV_SUMMARY = RESULTS / "cross_validation" / "cv_summary.csv"
CV_AGGREGATE = RESULTS / "cross_validation" / "cv_aggregate.json"

OUT_DIR = ROOT / "figures" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONVERGENCE = OUT_DIR / "stage1_cv_convergence.pdf"
OUT_REPRESENTATION = OUT_DIR / "stage1_representation_quality.pdf"
OUT_REPRESENTATIVE = OUT_DIR / "stage1_representative_fold.pdf"
OUT_MANIFEST = OUT_DIR / "stage1_figure_manifest.json"


# ============================================================
# PUBLICATION PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Vector-compatible PDF text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Export quality
    "figure.dpi": 150,
    "savefig.dpi": 600,

    # Clean axes
    "axes.spines.top": True,
    "axes.spines.right": True,
})


# ============================================================
# LOAD CROSS-VALIDATION SUMMARY
# ============================================================

if not CV_SUMMARY.exists():
    raise FileNotFoundError(
        f"Missing CV summary:\n{CV_SUMMARY}"
    )

if not CV_AGGREGATE.exists():
    raise FileNotFoundError(
        f"Missing CV aggregate:\n{CV_AGGREGATE}"
    )

cv = pd.read_csv(CV_SUMMARY)

with open(CV_AGGREGATE, "r", encoding="utf-8") as f:
    cv_aggregate = json.load(f)


# ------------------------------------------------------------
# Detect folds
# ------------------------------------------------------------

folds = sorted(cv["fold"].astype(int).tolist())

if len(folds) != 10:
    raise RuntimeError(
        f"Expected 10 folds, but found {len(folds)}:\n"
        f"{folds}"
    )

median_best_epoch = int(
    cv_aggregate["median_best_epoch"]
)


# ============================================================
# LOAD TRAINING HISTORY FOR EACH FOLD
# ============================================================

histories = {}

required_columns = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_positive_cosine",
    "val_negative_cosine",
    "val_cosine_gap",
    "val_embedding_std",
}

for fold in folds:

    history_path = (
        RESULTS
        / "folds"
        / f"fold_{fold:02d}"
        / "training_history.csv"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing training history:\n"
            f"{history_path}"
        )

    df = pd.read_csv(history_path)

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Fold {fold:02d} is missing columns:\n"
            f"{sorted(missing_columns)}"
        )

    df = df.sort_values("epoch").reset_index(drop=True)

    histories[fold] = df


# ============================================================
# TRAINING LENGTHS AND COMMON EPOCH INTERVAL
# ============================================================

max_epochs = {
    fold: int(df["epoch"].max())
    for fold, df in histories.items()
}

common_max_epoch = min(max_epochs.values())


print("\nTraining lengths:")

for fold in folds:
    print(
        f"  Fold {fold:02d}: "
        f"{max_epochs[fold]} epochs"
    )

print(
    f"\nCommon epoch range across all folds: "
    f"1--{common_max_epoch}"
)

print(
    f"Median selected best epoch: "
    f"{median_best_epoch}"
)


# ============================================================
# EXTRACT METRICS AT EACH SELECTED CHECKPOINT
# ============================================================

best_rows = []

for _, row in cv.iterrows():

    fold = int(row["fold"])
    best_epoch = int(row["best_epoch"])

    history = histories[fold]

    match = history.loc[
        history["epoch"].astype(int) == best_epoch
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Could not uniquely locate best epoch "
            f"{best_epoch} for Fold {fold:02d}"
        )

    h = match.iloc[0]

    best_rows.append({
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": float(
            row["best_val_loss"]
        ),
        "positive_cosine": float(
            h["val_positive_cosine"]
        ),
        "negative_cosine": float(
            h["val_negative_cosine"]
        ),
        "cosine_gap": float(
            h["val_cosine_gap"]
        ),
        "embedding_std": float(
            h["val_embedding_std"]
        ),
    })


best_df = (
    pd.DataFrame(best_rows)
    .sort_values("fold")
    .reset_index(drop=True)
)


# ============================================================
# SELECT REPRESENTATIVE FOLD
# ============================================================
#
# Representative fold is NOT the best fold.
#
# Selection criterion:
# robust normalized distance to the cross-fold median of:
#
#   1. best validation loss
#   2. cosine gap
#   3. embedding standard deviation
#   4. selected best epoch
#
# Median absolute deviation (MAD) is used for normalization.
# ============================================================

representative_metrics = [
    "best_val_loss",
    "cosine_gap",
    "embedding_std",
    "best_epoch",
]

medians = (
    best_df[representative_metrics]
    .median()
)

mad = (
    best_df[representative_metrics]
    .sub(medians)
    .abs()
    .median()
)

# Prevent division by zero
mad = mad.replace(0, 1e-12)

distance = (
    best_df[representative_metrics]
    .sub(medians)
    .abs()
    .div(mad)
    .mean(axis=1)
)

best_df["representative_distance"] = distance

representative_row = (
    best_df
    .sort_values(
        ["representative_distance", "fold"]
    )
    .iloc[0]
)

representative_fold = int(
    representative_row["fold"]
)

print(
    f"\nRepresentative fold selected "
    f"by median-distance rule: "
    f"Fold {representative_fold:02d}"
)


# ============================================================
# PREPARE CROSS-FOLD TRAINING MATRICES
# ============================================================

epochs = np.arange(
    1,
    common_max_epoch + 1
)

train_matrix = []
val_matrix = []

for fold in folds:

    df = histories[fold]

    common_df = (
        df[
            df["epoch"]
            <= common_max_epoch
        ]
        .sort_values("epoch")
    )

    if len(common_df) != common_max_epoch:
        raise RuntimeError(
            f"Fold {fold:02d} does not contain "
            f"all epochs from 1 to "
            f"{common_max_epoch}."
        )

    train_matrix.append(
        common_df["train_loss"].to_numpy()
    )

    val_matrix.append(
        common_df["val_loss"].to_numpy()
    )


train_matrix = np.vstack(train_matrix)
val_matrix = np.vstack(val_matrix)


train_mean = train_matrix.mean(axis=0)
train_std = train_matrix.std(
    axis=0,
    ddof=1
)

val_mean = val_matrix.mean(axis=0)
val_std = val_matrix.std(
    axis=0,
    ddof=1
)


# ============================================================
# FIGURE 1
# STAGE-1 CROSS-VALIDATION CONVERGENCE
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10.8, 4.15),
    constrained_layout=True
)


# ------------------------------------------------------------
# Panel (a): mean training / validation convergence
# ------------------------------------------------------------

ax = axes[0]

# Training mean
ax.plot(
    epochs,
    train_mean,
    linewidth=1.8,
    label="Training loss"
)

# Training ± SD
ax.fill_between(
    epochs,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.16
)

# Validation mean
ax.plot(
    epochs,
    val_mean,
    linewidth=1.9,
    label="Validation loss"
)

# Validation ± SD
ax.fill_between(
    epochs,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.16
)

# Zero reference
ax.axhline(
    y=0,
    linestyle="--",
    linewidth=0.8,
    alpha=0.55
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Self-supervised loss")

ax.set_title(
    "(a) Cross-validation convergence"
)

ax.set_xlim(
    1,
    common_max_epoch
)

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="upper right"
)

# IMPORTANT:
# No explanatory text is placed inside the curve area.
# Mean ± SD and common epoch range are explained
# in the LaTeX figure caption.


# ------------------------------------------------------------
# Panel (b): selected best epoch per fold
# ------------------------------------------------------------

ax = axes[1]

x = best_df["fold"].to_numpy()
best_epochs = best_df[
    "best_epoch"
].to_numpy()

ax.plot(
    x,
    best_epochs,
    marker="o",
    markersize=6,
    linewidth=1.2,
    alpha=0.75
)

ax.scatter(
    x,
    best_epochs,
    s=42,
    zorder=3
)

ax.axhline(
    y=median_best_epoch,
    linestyle="--",
    linewidth=1.4,
    label=(
        f"Median = "
        f"{median_best_epoch}"
    )
)

ax.set_xticks(x)

ax.set_xlabel(
    "Cross-validation fold"
)

ax.set_ylabel(
    "Selected best epoch"
)

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
    loc="upper left"
)


# ------------------------------------------------------------
# Save Figure 1
# ------------------------------------------------------------

fig.savefig(
    OUT_CONVERGENCE,
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2
# STAGE-1 REPRESENTATION QUALITY
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(12.6, 3.95),
    constrained_layout=True
)

x = best_df["fold"].to_numpy()


# ------------------------------------------------------------
# Panel (a): positive vs negative cosine similarity
# ------------------------------------------------------------

ax = axes[0]

positive = best_df[
    "positive_cosine"
].to_numpy()

negative = best_df[
    "negative_cosine"
].to_numpy()

ax.plot(
    x,
    positive,
    marker="o",
    markersize=5,
    linewidth=1.4,
    label="Positive pairs"
)

ax.plot(
    x,
    negative,
    marker="s",
    markersize=5,
    linewidth=1.4,
    label="Negative pairs"
)

ax.set_xticks(x)

ax.set_ylim(
    -0.05,
    1.05
)

ax.set_xlabel("Fold")
ax.set_ylabel("Cosine similarity")

ax.set_title(
    "(a) Pairwise cosine similarity"
)

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="center right"
)


# ------------------------------------------------------------
# Panel (b): cosine gap
# ------------------------------------------------------------

ax = axes[1]

cosine_gap = best_df[
    "cosine_gap"
].to_numpy()

gap_mean = cosine_gap.mean()
gap_std = cosine_gap.std(ddof=1)

ax.plot(
    x,
    cosine_gap,
    marker="o",
    markersize=5,
    linewidth=1.2,
    alpha=0.75
)

ax.scatter(
    x,
    cosine_gap,
    s=38,
    zorder=3
)

ax.axhline(
    y=gap_mean,
    linestyle="--",
    linewidth=1.4,
    label=(
        f"Mean = {gap_mean:.4f}"
    )
)

ax.set_xticks(x)

# Narrow range intentionally displays
# small fold-to-fold variation.
gap_padding = max(
    0.001,
    3.0 * gap_std
)

ax.set_ylim(
    gap_mean - gap_padding,
    gap_mean + gap_padding
)

ax.set_xlabel("Fold")
ax.set_ylabel("Cosine gap")

ax.set_title(
    "(b) Representation separation"
)

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.text(
    0.03,
    0.97,
    (
        f"Mean ± SD\n"
        f"{gap_mean:.4f} ± "
        f"{gap_std:.4f}"
    ),
    transform=ax.transAxes,
    fontsize=8.3,
    va="top",
    ha="left"
)


# ------------------------------------------------------------
# Panel (c): embedding standard deviation
# ------------------------------------------------------------

ax = axes[2]

embedding_std = best_df[
    "embedding_std"
].to_numpy()

emb_mean = embedding_std.mean()
emb_std_sd = embedding_std.std(
    ddof=1
)

ax.plot(
    x,
    embedding_std,
    marker="o",
    markersize=5,
    linewidth=1.2,
    alpha=0.75
)

ax.scatter(
    x,
    embedding_std,
    s=38,
    zorder=3
)

ax.axhline(
    y=emb_mean,
    linestyle="--",
    linewidth=1.4,
    label=(
        f"Mean = {emb_mean:.4f}"
    )
)

ax.set_xticks(x)

ax.set_xlabel("Fold")
ax.set_ylabel("Embedding SD")

ax.set_title(
    "(c) Embedding variability"
)

ax.grid(
    alpha=0.18,
    linewidth=0.6
)

ax.legend(
    frameon=False,
    loc="lower right"
)

ax.text(
    0.03,
    0.97,
    (
        f"Mean ± SD\n"
        f"{emb_mean:.5f} ± "
        f"{emb_std_sd:.5f}"
    ),
    transform=ax.transAxes,
    fontsize=8.3,
    va="top",
    ha="left"
)


# ------------------------------------------------------------
# Save Figure 2
# ------------------------------------------------------------

fig.savefig(
    OUT_REPRESENTATION,
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3
# REPRESENTATIVE FOLD DIAGNOSTICS
# ============================================================

representative_plot_dir = (
    RESULTS
    / "folds"
    / f"fold_{representative_fold:02d}"
    / "plots"
)

loss_png = (
    representative_plot_dir
    / "loss_curves.png"
)

diagnostic_png = (
    representative_plot_dir
    / "representation_diagnostics.png"
)


if not loss_png.exists():
    raise FileNotFoundError(
        f"Missing representative loss plot:\n"
        f"{loss_png}"
    )

if not diagnostic_png.exists():
    raise FileNotFoundError(
        f"Missing representative diagnostic plot:\n"
        f"{diagnostic_png}"
    )


loss_img = Image.open(
    loss_png
).convert("RGB")

diagnostic_img = Image.open(
    diagnostic_png
).convert("RGB")


fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12.2, 4.4),
    constrained_layout=True
)


# ------------------------------------------------------------
# Left: original representative loss curves
# ------------------------------------------------------------

axes[0].imshow(loss_img)

axes[0].axis("off")

axes[0].set_title(
    "(a) Training and validation diagnostics"
)


# ------------------------------------------------------------
# Right: original representation diagnostics
# ------------------------------------------------------------

axes[1].imshow(diagnostic_img)

axes[1].axis("off")

axes[1].set_title(
    "(b) Representation diagnostics"
)


fig.suptitle(
    (
        f"Representative Stage-1 fold: "
        f"Fold {representative_fold:02d}"
    ),
    fontsize=12
)


# ------------------------------------------------------------
# Save Figure 3
# ------------------------------------------------------------

fig.savefig(
    OUT_REPRESENTATIVE,
    format="pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# CALCULATE FINAL NUMERICAL SUMMARY
# ============================================================

best_val_mean = best_df[
    "best_val_loss"
].mean()

best_val_sd = best_df[
    "best_val_loss"
].std(ddof=1)

positive_mean = best_df[
    "positive_cosine"
].mean()

positive_sd = best_df[
    "positive_cosine"
].std(ddof=1)

negative_mean = best_df[
    "negative_cosine"
].mean()

negative_sd = best_df[
    "negative_cosine"
].std(ddof=1)

gap_mean = best_df[
    "cosine_gap"
].mean()

gap_sd = best_df[
    "cosine_gap"
].std(ddof=1)

embedding_mean = best_df[
    "embedding_std"
].mean()

embedding_sd = best_df[
    "embedding_std"
].std(ddof=1)

best_epoch_min = int(
    best_df["best_epoch"].min()
)

best_epoch_max = int(
    best_df["best_epoch"].max()
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "source_results_directory": str(
        RESULTS
    ),

    "number_of_folds": len(folds),

    "fold_training_lengths": {
        f"fold_{fold:02d}": int(
            max_epochs[fold]
        )
        for fold in folds
    },

    "common_epoch_interval": {
        "start": 1,
        "end": int(common_max_epoch)
    },

    "cross_validation_summary": {

        "best_validation_loss": {
            "mean": float(
                best_val_mean
            ),
            "sd": float(
                best_val_sd
            )
        },

        "positive_cosine_similarity": {
            "mean": float(
                positive_mean
            ),
            "sd": float(
                positive_sd
            )
        },

        "negative_cosine_similarity": {
            "mean": float(
                negative_mean
            ),
            "sd": float(
                negative_sd
            )
        },

        "cosine_gap": {
            "mean": float(
                gap_mean
            ),
            "sd": float(
                gap_sd
            )
        },

        "embedding_standard_deviation": {
            "mean": float(
                embedding_mean
            ),
            "sd": float(
                embedding_sd
            )
        },

        "median_best_epoch": int(
            median_best_epoch
        ),

        "best_epoch_range": [
            best_epoch_min,
            best_epoch_max
        ]
    },

    "representative_fold": {
        "fold": (
            f"fold_{representative_fold:02d}"
        ),
        "selection_rule": (
            "Minimum robust normalized distance "
            "to the cross-fold median of best "
            "validation loss, cosine gap, "
            "embedding standard deviation, "
            "and selected best epoch."
        )
    },

    "outputs": {
        "stage1_cv_convergence": str(
            OUT_CONVERGENCE
        ),
        "stage1_representation_quality": str(
            OUT_REPRESENTATION
        ),
        "stage1_representative_fold": str(
            OUT_REPRESENTATIVE
        )
    },

    "best_checkpoint_metrics": (
        best_df.to_dict(
            orient="records"
        )
    )
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

print("\n" + "=" * 78)
print("STAGE-1 PUBLICATION FIGURES CREATED")
print("=" * 78)

print(
    f"\n1. {OUT_CONVERGENCE}"
)

print(
    f"2. {OUT_REPRESENTATION}"
)

print(
    f"3. {OUT_REPRESENTATIVE}"
)

print(
    f"4. {OUT_MANIFEST}"
)


print("\nCross-validation summary:")

print(
    f"  Best validation loss: "
    f"{best_val_mean:.6f} "
    f"± "
    f"{best_val_sd:.6f}"
)

print(
    f"  Positive cosine: "
    f"{positive_mean:.6f} "
    f"± "
    f"{positive_sd:.6f}"
)

print(
    f"  Negative cosine: "
    f"{negative_mean:.6f} "
    f"± "
    f"{negative_sd:.6f}"
)

print(
    f"  Cosine gap: "
    f"{gap_mean:.6f} "
    f"± "
    f"{gap_sd:.6f}"
)

print(
    f"  Embedding SD: "
    f"{embedding_mean:.6f} "
    f"± "
    f"{embedding_sd:.6f}"
)

print(
    f"  Median best epoch: "
    f"{median_best_epoch}"
)

print(
    f"  Best-epoch range: "
    f"{best_epoch_min}--"
    f"{best_epoch_max}"
)

print(
    f"  Common CV epoch range: "
    f"1--{common_max_epoch}"
)

print(
    f"  Representative fold: "
    f"fold_{representative_fold:02d}"
)

print("\n" + "=" * 78)
