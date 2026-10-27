"""Train a traffic-sign (GTSRB) or traffic-light (red/yellow/green) classifier."""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from traffic_ai.data import (
    FolderImageDataset,
    TorchvisionGTSRBAdapter,
    TransformView,
    limit_per_class,
    resolve_folder_root,
    split_train_validation,
    stratified_split,
)
from traffic_ai.models import build_model
from traffic_ai.vision import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    calculate_metrics,
    make_transforms,
    save_history,
    save_metric_artifacts,
    save_misclassified,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN, MobileNetV3-Small or ResNet18.")
    parser.add_argument("--dataset", choices=["gtsrb", "traffic_lights"], default="gtsrb",
                        help="gtsrb=43 traffic-sign classes; traffic_lights=red/yellow/green folders")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional local dataset folder. If omitted for GTSRB, torchvision downloads it.")
    parser.add_argument("--data-dir", type=str, default="data", help="Download/cache directory for GTSRB")
    parser.add_argument("--model", choices=["cnn", "mobilenet_v3_small", "resnet18"], default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None, help="Default: 0.001 for CNN, 0.0003 for transfer learning")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience in epochs")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Used for local folder datasets; torchvision GTSRB uses its official test split")
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Optional cap for faster experiments; 0 uses every image")
    parser.add_argument("--hflip", type=float, default=None,
                        help="Horizontal-flip probability. Default: 0 for signs, 0.5 for traffic lights.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the transfer-learning head")
    parser.add_argument("--class-weighting", action="store_true", help="Use inverse-square-root class weights")
    parser.add_argument("--skip-corrupt", action="store_true",
                        help="Verify local image files first and skip files that cannot be decoded")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for the most reliable Windows experience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output directory for checkpoint and reports")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]
    raise ValueError("Dataset adapter does not expose labels")


def build_data(args):
    """Returns train/val/test base datasets, indices, and class names."""
    if args.dataset == "gtsrb":
        if args.data:
            root = resolve_folder_root(args.data, "gtsrb")
            base = FolderImageDataset(root, "gtsrb", skip_corrupt=args.skip_corrupt)
            if base.corrupt_files:
                print(f"Skipped {len(base.corrupt_files)} unreadable images.")
            labels = get_labels(base)
            selected = limit_per_class(list(range(len(base))), labels, args.max_per_class, args.seed)
            tr, va, te = stratified_split(selected, labels, args.val_ratio, args.test_ratio, args.seed)
            return base, tr, base, va, base, te, base.classes, "random_stratified"

        print("Loading GTSRB through torchvision (first run may download the dataset)...")
        try:
            raw_train = datasets.GTSRB(root=args.data_dir, split="train", download=True, transform=None)
            raw_test = datasets.GTSRB(root=args.data_dir, split="test", download=True, transform=None)
        except Exception as exc:
            raise RuntimeError(
                "GTSRB could not be downloaded automatically. Download/unpack GTSRB to class folders, "
                "then pass --data PATH. See README_FA.md for the expected layouts.\n"
                f"Original error: {exc}"
            ) from exc
        train_base = TorchvisionGTSRBAdapter(raw_train)
        test_base = TorchvisionGTSRBAdapter(raw_test)
        labels = get_labels(train_base)
        selected = limit_per_class(list(range(len(train_base))), labels, args.max_per_class, args.seed)
        tr, va = split_train_validation(selected, labels, args.val_ratio, args.seed)
        test_ids = list(range(len(test_base)))
        return train_base, tr, train_base, va, test_base, test_ids, train_base.classes, "official_test"

    if not args.data:
        raise ValueError(
            "For --dataset traffic_lights, supply --data pointing to a folder containing "
            "red/, yellow/, and green/ subfolders. GTSRB does not contain traffic-light labels."
        )
    root = resolve_folder_root(args.data, "traffic_lights")
    base = FolderImageDataset(root, "traffic_lights", skip_corrupt=args.skip_corrupt)
    if base.corrupt_files:
        print(f"Skipped {len(base.corrupt_files)} unreadable images.")
    labels = get_labels(base)
    selected = limit_per_class(list(range(len(base))), labels, args.max_per_class, args.seed)
    tr, va, te = stratified_split(selected, labels, args.val_ratio, args.test_ratio, args.seed)
    return base, tr, base, va, base, te, base.classes, "random_stratified"


def make_loader(base, indices, image_size, train, batch_size, workers, device, horizontal_flip):
    transform = make_transforms(image_size, train, horizontal_flip if train else 0.0)
    view = TransformView(base, indices, transform)
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    correct = 0
    amp_enabled = device.type == "cuda"

    iterator = tqdm(loader, desc="train" if training else "eval", leave=False)
    for images, labels, _sample_ids in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        batch = labels.size(0)
        total_loss += float(loss.detach().item()) * batch
        total_count += batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        iterator.set_postfix(loss=f"{total_loss / max(total_count, 1):.4f}",
                             acc=f"{correct / max(total_count, 1):.3f}")
    return total_loss / max(total_count, 1), correct / max(total_count, 1)


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[int] = []
    confidences: list[float] = []
    for images, labels, ids in tqdm(loader, desc="test", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        confidence, predictions = probs.max(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.cpu().tolist())
        sample_ids.extend(ids.tolist())
        confidences.extend(confidence.cpu().tolist())
    return y_true, y_pred, sample_ids, confidences


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.img_size < 32:
        raise SystemExit("epochs/batch-size must be positive and img-size must be at least 32")
    if not 0 < args.val_ratio < 0.5 or not 0 < args.test_ratio < 0.5 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("val-ratio and test-ratio must be positive and sum to less than 1")
    if args.hflip is not None and not 0 <= args.hflip <= 1:
        raise SystemExit("--hflip must be in [0, 1]")

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    horizontal_flip = args.hflip if args.hflip is not None else (0.0 if args.dataset == "gtsrb" else 0.5)
    output_dir = Path(args.output) if args.output else Path("runs") / (
        f"{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Horizontal flip: {horizontal_flip}")
    train_base, train_ids, val_base, val_ids, test_base, test_ids, class_names, split_type = build_data(args)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {', '.join(class_names[:10])}{' ...' if num_classes > 10 else ''}")
    print(f"Samples — train: {len(train_ids):,}, validation: {len(val_ids):,}, test: {len(test_ids):,} ({split_type})")

    train_loader = make_loader(train_base, train_ids, args.img_size, True, args.batch_size,
                               args.workers, device, horizontal_flip)
    val_loader = make_loader(val_base, val_ids, args.img_size, False, args.batch_size,
                             args.workers, device, 0.0)
    test_loader = make_loader(test_base, test_ids, args.img_size, False, args.batch_size,
                              args.workers, device, 0.0)

    model = build_model(args.model, num_classes, pretrained=not args.no_pretrained,
                        freeze_backbone=args.freeze_backbone).to(device)
    lr = args.lr if args.lr is not None else (1e-3 if args.model == "cnn" else 3e-4)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    loss_weights = None
    if args.class_weighting:
        counts = np.bincount([int(train_base.targets[i]) for i in train_ids], minlength=num_classes).astype(np.float32)
        inverse_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        inverse_sqrt = inverse_sqrt / inverse_sqrt.mean()
        loss_weights = torch.tensor(inverse_sqrt, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = vars(args).copy()
    config.update({"device": str(device), "class_names": class_names, "horizontal_flip": horizontal_flip,
                   "split_type": split_type, "train_samples": len(train_ids),
                   "validation_samples": len(val_ids), "test_samples": len(test_ids),
                   "imagenet_mean": IMAGENET_MEAN, "imagenet_std": IMAGENET_STD})
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        row = {"epoch": float(epoch), "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        print(f"train loss={train_loss:.4f}, acc={train_acc:.4f} | val loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "dataset": args.dataset,
                "class_names": class_names,
                "num_classes": num_classes,
                "image_size": args.img_size,
                "imagenet_mean": IMAGENET_MEAN,
                "imagenet_std": IMAGENET_STD,
                "pretrained_requested": not args.no_pretrained,
                "freeze_backbone": args.freeze_backbone,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
            }, checkpoint_path)
            print(f"Saved new best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        save_history(history, output_dir)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: validation accuracy did not improve for {args.patience} epochs.")
            break

    elapsed_seconds = time.time() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_true, test_pred, test_ids_seen, confidences = predict_loader(model, test_loader, device)
    metrics = calculate_metrics(test_true, test_pred, class_names)
    metrics.update({
        "dataset": args.dataset,
        "model": args.model,
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_val_acc),
        "test_samples": len(test_true),
        "training_seconds": round(elapsed_seconds, 2),
        "device": str(device),
        "split_type": split_type,
    })
    save_metric_artifacts(metrics, output_dir, class_names, test_true, test_pred)

    mistakes = []
    for true_label, pred_label, index, confidence in zip(test_true, test_pred, test_ids_seen, confidences):
        if true_label != pred_label:
            mistakes.append({"index": int(index), "true": int(true_label), "pred": int(pred_label),
                             "confidence": float(confidence)})
    save_misclassified(test_base, mistakes, output_dir, class_names)

    print("\n=== TEST RESULTS ===")
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"):
        print(f"{key:20s}: {metrics[key]:.4f}")
    print(f"Best epoch: {best_epoch} | Test samples: {len(test_true):,} | Runtime: {elapsed_seconds / 60:.1f} min")
    print(f"\nAll outputs saved to: {output_dir.resolve()}")
    print("Files: best_model.pth, metrics.json, classification_report.json, confusion_matrix*.png, "
          "loss_curve.png, accuracy_curve.png, history.csv, misclassified_examples*.\n")


if __name__ == "__main__":
    main()
