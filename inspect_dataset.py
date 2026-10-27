"""Audit a local folder-labelled image dataset and generate summary plots."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from traffic_ai.data import FolderImageDataset, resolve_folder_root


def main():
    parser = argparse.ArgumentParser(description="Inspect local GTSRB or traffic-light images before training.")
    parser.add_argument("--dataset", choices=["gtsrb", "traffic_lights"], required=True)
    parser.add_argument("--data", required=True, help="Path to folder containing labelled subfolders")
    parser.add_argument("--output", default="dataset_audit", help="Where to save audit report and plots")
    parser.add_argument("--sample-count", type=int, default=20, help="Maximum images for preview grid")
    parser.add_argument("--check-images", action="store_true", help="Actually decode every file to find corruption")
    args = parser.parse_args()

    root = resolve_folder_root(args.data, args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = FolderImageDataset(root, args.dataset, skip_corrupt=args.check_images)
    counts = Counter(dataset.targets)
    class_names = dataset.classes
    dimensions: list[tuple[int, int]] = []
    corrupt_files: list[str] = list(dataset.corrupt_files)
    valid_indices: list[int] = []

    # If file checking is off, this loop still gathers dimensions while reporting any files
    # that fail during normal image decoding.
    for idx, path_str in enumerate(dataset.paths):
        try:
            with Image.open(path_str) as image:
                image.verify()
            with Image.open(path_str) as image:
                dimensions.append((int(image.width), int(image.height)))
            valid_indices.append(idx)
        except Exception:
            corrupt_files.append(path_str)

    report = {
        "dataset": args.dataset,
        "root": str(root.resolve()),
        "classes": len(class_names),
        "total_images_listed": len(dataset.samples),
        "valid_images_checked": len(valid_indices),
        "corrupt_images": len(set(corrupt_files)),
        "class_distribution": {class_names[i]: int(counts.get(i, 0)) for i in range(len(class_names))},
        "width_min": min((w for w, _h in dimensions), default=None),
        "width_median": float(np.median([w for w, _h in dimensions])) if dimensions else None,
        "width_max": max((w for w, _h in dimensions), default=None),
        "height_min": min((h for _w, h in dimensions), default=None),
        "height_median": float(np.median([h for _w, h in dimensions])) if dimensions else None,
        "height_max": max((h for _w, h in dimensions), default=None),
        "corrupt_files": sorted(set(corrupt_files)),
    }
    (output / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = list(range(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(9, len(class_names) * 0.35), 5))
    values = [counts.get(i, 0) for i in labels]
    ax.bar(labels, values)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Keep preview compact and representative. At most one image per class initially.
    rng = random.Random(42)
    per_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        per_class[int(label)].append(idx)
    preview_ids = []
    for label in sorted(per_class):
        if per_class[label]:
            preview_ids.append(rng.choice(per_class[label]))
    rng.shuffle(preview_ids)
    preview_ids = preview_ids[:max(1, args.sample_count)]
    cols = min(5, max(1, len(preview_ids)))
    rows = (len(preview_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, preview_ids):
        try:
            image = dataset.get_image(idx)
            ax.imshow(image)
            class_id = dataset.targets[idx]
            label = class_names[class_id]
            ax.set_title(f"{class_id}: {label}", fontsize=7)
            ax.axis("off")
        except Exception:
            continue
    fig.suptitle("Sample images by class", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "sample_images.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset root: {root}")
    print(f"Classes: {len(class_names)} | Images found: {len(dataset.samples)}")
    print(f"Image sizes (width × height): min={report['width_min']}×{report['height_min']}, "
          f"median={report['width_median']}×{report['height_median']}, "
          f"max={report['width_max']}×{report['height_max']}")
    print(f"Unreadable images: {report['corrupt_images']}")
    print(f"Saved audit: {(output / 'dataset_audit.json').resolve()}")
    print(f"Saved plots: {(output / 'class_distribution.png').resolve()}, {(output / 'sample_images.png').resolve()}")
    if corrupt_files:
        print("Inspect dataset_audit.json for corrupt file paths. During training, use --skip-corrupt to skip them.")


if __name__ == "__main__":
    main()
