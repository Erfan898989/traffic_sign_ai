"""Training transforms, metrics, plots, and checkpoint helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(image_size: int, training: bool, horizontal_flip: float = 0.0):
    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if training:
        aug = [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.035),
        ]
        if horizontal_flip > 0:
            aug.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
        aug += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(aug)
    return transforms.Compose([transforms.Resize((image_size, image_size)), *normalize])


def calculate_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_metric_artifacts(metrics: dict[str, Any], output_dir: Path, class_names: list[str],
                          y_true: list[int], y_pred: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = metrics["report"]
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        clean = {k: v for k, v in metrics.items() if k not in {"report", "confusion_matrix"}}
        json.dump(clean, f, ensure_ascii=False, indent=2)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    fig_size = max(8, min(18, 0.32 * len(class_names) + 5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label", title="Confusion Matrix",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=max(5, 10 - len(class_names) // 8))
    ax.tick_params(axis="y", labelsize=max(5, 10 - len(class_names) // 8))
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Normalized confusion matrix is useful when the classes are imbalanced.
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label", title="Normalized Confusion Matrix",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=max(5, 10 - len(class_names) // 8))
    ax.tick_params(axis="y", labelsize=max(5, 10 - len(class_names) // 8))
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    if not history:
        return
    import csv
    keys = list(history[0].keys())
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [row["train_accuracy"] for row in history], label="Train accuracy")
    ax.plot(epochs, [row["val_accuracy"] for row in history], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close(fig)


def save_misclassified(base_dataset, mistakes: list[dict[str, Any]], output_dir: Path,
                       class_names: list[str], limit: int = 25) -> None:
    if not mistakes:
        return
    mistakes = mistakes[:limit]
    cols = 5
    rows = (len(mistakes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, item in zip(axes, mistakes):
        image = base_dataset.get_image(item["index"])
        ax.imshow(image)
        true_name = class_names[item["true"]]
        pred_name = class_names[item["pred"]]
        ax.set_title(f"True: {true_name}\nPred: {pred_name} ({item['confidence']:.1%})", fontsize=8)
        ax.axis("off")
    fig.suptitle("Misclassified test examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "misclassified_examples.csv").open("w", encoding="utf-8") as f:
        f.write("index,true_label,predicted_label,confidence,image_path\n")
        for item in mistakes:
            path = getattr(base_dataset, "paths", [""] * (item["index"] + 1))[item["index"]]
            f.write(f"{item['index']},{class_names[item['true']]},{class_names[item['pred']]},{item['confidence']:.6f},\"{path}\"\n")


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location=device)
