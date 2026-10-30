from pathlib import Path


def save_loss_plot(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [r["train_total_loss"] for r in history],
        label="Train subject-balanced total",
    )
    plt.plot(
        epochs,
        [r["val_total_loss"] for r in history],
        label="Validation subject-balanced total",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Stage-2 loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_metric_plot(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(11, 6))
    for key, label in [
        ("val_pearson_r", "Pearson r"),
        ("val_spectral_corr", "Spectral corr"),
        ("val_std_ratio", "Pred/target std ratio"),
        ("val_best_lag_r", "Best-lag r (diagnostic)"),
    ]:
        if key in history[0]:
            plt.plot(epochs, [r[key] for r in history], label=label)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_waveform_examples(examples, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples),
        1,
        figsize=(12, 2.6 * len(examples)),
        squeeze=False,
    )

    for ax, ex in zip(axes[:, 0], examples):
        ax.plot(ex["target"], label="Ground-truth normalized PPG")
        ax.plot(ex["pred"], label="Predicted normalized rPPG")
        ax.set_title(ex["title"])
        ax.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
