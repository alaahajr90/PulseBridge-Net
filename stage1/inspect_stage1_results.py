from pathlib import Path
from PIL import Image
import csv
import json
import os
import re

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/home/hajr/mcd_rppg/code_HR_RR_model/"
    "PulseBridge_Net/Stage_1_PulseRepresentation/results"
)

OUT_DIR = ROOT / "_publication_inventory"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREE_TXT = OUT_DIR / "stage1_results_tree.txt"
IMAGE_CSV = OUT_DIR / "stage1_image_inventory.csv"
DATA_CSV = OUT_DIR / "stage1_data_inventory.csv"
GROUPS_JSON = OUT_DIR / "stage1_plot_groups.json"


# ============================================================
# HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"
}

DATA_EXTENSIONS = {
    ".csv", ".json", ".txt", ".npy", ".npz", ".pt", ".pth"
}


def get_fold_name(path: Path):
    """
    Detect fold_01, fold_02, ..., fold_10 from the path.
    """
    for part in path.parts:
        if re.fullmatch(r"fold_\d+", part):
            return part
    return ""


def human_size(num_bytes):
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    return f"{kb / 1024:.2f} MB"


def classify_plot(filename: str):
    """
    Rough grouping only for inventory purposes.
    We will decide the final publication grouping manually afterward.
    """
    name = filename.lower()

    if "loss" in name or "curve" in name or "history" in name:
        return "training_convergence"

    if (
        "representation" in name
        or "cosine" in name
        or "embedding" in name
        or "diagnostic" in name
    ):
        return "representation_quality"

    if "similarity" in name:
        return "representation_quality"

    if "summary" in name:
        return "summary"

    return "other"


# ============================================================
# 1. DIRECTORY TREE
# ============================================================

def build_tree(root: Path):
    lines = []

    def walk(directory: Path, prefix=""):
        entries = sorted(
            directory.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "

            try:
                rel = entry.relative_to(root)
            except ValueError:
                rel = entry

            lines.append(prefix + connector + entry.name)

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension)

    lines.append(root.name + "/")
    walk(root)

    return "\n".join(lines)


# ============================================================
# 2. IMAGE INVENTORY
# ============================================================

def inspect_images(root: Path):
    rows = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()

        if ext not in IMAGE_EXTENSIONS:
            continue

        # Skip our inventory folder
        if OUT_DIR in path.parents:
            continue

        fold = get_fold_name(path)

        width = ""
        height = ""
        aspect_ratio = ""
        dpi_x = ""
        dpi_y = ""
        mode = ""
        image_format = ext.replace(".", "").upper()

        # Pillow does not inspect vector PDF dimensions reliably,
        # so PDF metadata is kept without pixel dimensions.
        if ext != ".pdf":
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    mode = img.mode
                    image_format = img.format or image_format

                    if height:
                        aspect_ratio = width / height

                    dpi = img.info.get("dpi", None)

                    if dpi:
                        if isinstance(dpi, tuple):
                            if len(dpi) >= 2:
                                dpi_x = dpi[0]
                                dpi_y = dpi[1]
                        else:
                            dpi_x = dpi
                            dpi_y = dpi

            except Exception as exc:
                print(f"[WARNING] Could not inspect {path}: {exc}")

        file_size = path.stat().st_size

        rows.append({
            "fold": fold,
            "group": classify_plot(path.name),
            "filename": path.name,
            "extension": ext,
            "format": image_format,
            "width_px": width,
            "height_px": height,
            "aspect_ratio": (
                f"{aspect_ratio:.4f}" if aspect_ratio != "" else ""
            ),
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
            "color_mode": mode,
            "size_bytes": file_size,
            "size_human": human_size(file_size),
            "relative_path": str(path.relative_to(root)),
            "absolute_path": str(path),
        })

    return rows


# ============================================================
# 3. DATA / RESULT FILE INVENTORY
# ============================================================

def inspect_data_files(root: Path):
    rows = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if OUT_DIR in path.parents:
            continue

        ext = path.suffix.lower()

        if ext not in DATA_EXTENSIONS:
            continue

        fold = get_fold_name(path)
        size = path.stat().st_size

        rows.append({
            "fold": fold,
            "filename": path.name,
            "extension": ext,
            "size_bytes": size,
            "size_human": human_size(size),
            "relative_path": str(path.relative_to(root)),
            "absolute_path": str(path),
        })

    return rows


# ============================================================
# 4. WRITE CSV
# ============================================================

def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 5. GROUP PLOTS
# ============================================================

def build_groups(image_rows):
    groups = {}

    for row in image_rows:
        group = row["group"]
        groups.setdefault(group, [])

        groups[group].append({
            "fold": row["fold"],
            "filename": row["filename"],
            "width_px": row["width_px"],
            "height_px": row["height_px"],
            "aspect_ratio": row["aspect_ratio"],
            "relative_path": row["relative_path"],
        })

    return groups


# ============================================================
# 6. SUMMARY
# ============================================================

def print_summary(image_rows, data_rows):
    print("\n" + "=" * 80)
    print("STAGE-1 RESULTS INVENTORY")
    print("=" * 80)

    print(f"\nRoot:")
    print(ROOT)

    print(f"\nTotal image/figure files: {len(image_rows)}")
    print(f"Total data/result files:  {len(data_rows)}")

    folds = sorted(
        set(row["fold"] for row in image_rows + data_rows if row["fold"])
    )

    print(f"\nDetected folds ({len(folds)}):")
    for fold in folds:
        print(f"  - {fold}")

    print("\nImage groups:")

    counts = {}
    for row in image_rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1

    for group, count in sorted(counts.items()):
        print(f"  {group:30s}: {count}")

    print("\nUnique plot filenames across all folds:")

    unique_names = sorted(set(row["filename"] for row in image_rows))

    for name in unique_names:
        matching = [r for r in image_rows if r["filename"] == name]

        dimensions = sorted(
            set(
                (r["width_px"], r["height_px"])
                for r in matching
                if r["width_px"] != ""
            )
        )

        print(f"\n  {name}")
        print(f"    occurrences: {len(matching)}")

        if dimensions:
            print("    dimensions:")
            for w, h in dimensions:
                ratio = w / h if h else 0
                print(
                    f"      {w} x {h} px "
                    f"(aspect={ratio:.3f})"
                )

        folds_here = sorted(
            r["fold"] for r in matching if r["fold"]
        )

        if folds_here:
            print(f"    folds: {', '.join(folds_here)}")

    print("\nPotential numerical result files:")

    interesting = [
        row for row in data_rows
        if any(
            key in row["filename"].lower()
            for key in [
                "history",
                "summary",
                "metric",
                "result",
                "diagnostic",
                "cv",
                "fold"
            ]
        )
    ]

    for row in interesting:
        print(
            f"  [{row['fold'] or 'GLOBAL':8s}] "
            f"{row['relative_path']}"
        )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(TREE_TXT)
    print(IMAGE_CSV)
    print(DATA_CSV)
    print(GROUPS_JSON)


# ============================================================
# MAIN
# ============================================================

def main():
    if not ROOT.exists():
        raise FileNotFoundError(
            f"\nResults directory not found:\n{ROOT}\n"
        )

    # Directory tree
    tree = build_tree(ROOT)
    TREE_TXT.write_text(tree, encoding="utf-8")

    # Inventories
    image_rows = inspect_images(ROOT)
    data_rows = inspect_data_files(ROOT)

    write_csv(IMAGE_CSV, image_rows)
    write_csv(DATA_CSV, data_rows)

    # Plot grouping
    groups = build_groups(image_rows)

    GROUPS_JSON.write_text(
        json.dumps(groups, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Print useful terminal summary
    print_summary(image_rows, data_rows)


if __name__ == "__main__":
    main()
