from pathlib import Path

import numpy as np
import torch

from io_utils import save_json, write_rows_csv


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def grad_global_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5


def save_training_plots(history, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r.get("train_loss", np.nan) for r in history], label="Train loss")
    plt.plot(epochs, [r.get("val_loss", np.nan) for r in history], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("SSL loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_representation_plot(history, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [r.get("val_positive_cosine", np.nan) for r in history],
        label="Val positive cosine",
    )
    plt.plot(
        epochs,
        [r.get("val_negative_cosine", np.nan) for r in history],
        label="Val negative cosine",
    )
    plt.plot(
        epochs,
        [r.get("val_cosine_gap", np.nan) for r in history],
        label="Val cosine gap",
    )
    plt.plot(
        epochs,
        [r.get("val_embedding_std", np.nan) for r in history],
        label="Val embedding std",
    )
    plt.xlabel("Epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
