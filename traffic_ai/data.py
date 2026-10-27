"""Dataset utilities for GTSRB traffic signs and custom traffic-light images."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".pgm", ".tif", ".tiff"}
GTSRB_CLASS_NAMES = [
    "Speed limit (20 km/h)", "Speed limit (30 km/h)", "Speed limit (50 km/h)",
    "Speed limit (60 km/h)", "Speed limit (70 km/h)", "Speed limit (80 km/h)",
    "End speed limit (80 km/h)", "Speed limit (100 km/h)", "Speed limit (120 km/h)",
    "No passing", "No passing for vehicles over 3.5 t", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Straight or right",
    "Straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End no passing", "End no passing for vehicles over 3.5 t",
]


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _has_class_folders(root: Path, numeric: bool, required_light_classes: bool = False) -> bool:
    if not root.is_dir():
        return False
    folders = [p for p in root.iterdir() if p.is_dir()]
    if not folders:
        return False
    if required_light_classes:
        names = {p.name.lower() for p in folders}
        if not {"red", "yellow", "green"}.issubset(names):
            return False
    candidates = [p for p in folders if p.name.isdigit()] if numeric else folders
    return len(candidates) >= (3 if required_light_classes else 2) and any(_image_files(p) for p in candidates)


def resolve_folder_root(data_dir: str | Path, dataset: str) -> Path:
    """Resolve common Kaggle/GTSRB archive layouts to a class-folder root."""
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    if dataset == "traffic_lights":
        candidates = [root, root / "train", root / "Train", root / "images", root / "Images"]
        for candidate in candidates:
            if _has_class_folders(candidate, numeric=False, required_light_classes=True):
                return candidate
        raise ValueError(
            "Couldn't find red/yellow/green class folders. Expected a directory such as "
            "data/traffic_lights/{red,yellow,green}/image.jpg"
        )

    candidates = [
        root,
        root / "Train",
        root / "train",
        root / "Final_Training" / "Images",
        root / "GTSRB" / "Final_Training" / "Images",
        root / "GTSRB" / "Train",
        root / "images",
        root / "Images",
    ]
    for candidate in candidates:
        if _has_class_folders(candidate, numeric=True):
            # Avoid accidentally treating a parent folder with unrelated numbered folders as data.
            numeric_dirs = [p for p in candidate.iterdir() if p.is_dir() and p.name.isdigit()]
            if len(numeric_dirs) >= 2:
                return candidate
    raise ValueError(
        "Couldn't find GTSRB class folders. Point --data to the folder containing numbered "
        "class directories (0..42 or 00000..00042), or omit --data to use torchvision's download."
    )


class FolderImageDataset(Dataset):
    """Loads folder-labelled images without applying transforms, so each split can augment differently."""

    def __init__(self, root: str | Path, dataset: str, skip_corrupt: bool = False):
        self.root = Path(root)
        self.dataset = dataset
        self.skip_corrupt = skip_corrupt
        self.corrupt_files: list[str] = []
        class_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        class_dirs = [p for p in class_dirs if _image_files(p)]
        if dataset == "gtsrb":
            class_dirs = [p for p in class_dirs if p.name.isdigit()]
            class_dirs.sort(key=lambda p: int(p.name))
            max_class_id = max((int(p.name) for p in class_dirs), default=-1)
            self.classes = GTSRB_CLASS_NAMES[:max_class_id + 1] if max_class_id >= 0 else []
        else:
            # Normalize names but retain human-readable labels for predictions.
            class_dirs.sort(key=lambda p: p.name.lower())
            self.classes = [p.name.strip().lower().replace(" ", "_") for p in class_dirs]

        if not class_dirs:
            raise ValueError(f"No labelled image folders found under {self.root}")

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for label, class_dir in enumerate(class_dirs):
            actual_label = int(class_dir.name) if dataset == "gtsrb" else label
            for image_path in _image_files(class_dir):
                if self.skip_corrupt:
                    try:
                        with Image.open(image_path) as test_image:
                            test_image.verify()
                    except Exception:
                        self.corrupt_files.append(str(image_path))
                        continue
                self.samples.append((image_path, actual_label))
        self.targets = [label for _, label in self.samples]
        self.paths = [str(path) for path, _ in self.samples]

        if len(self.samples) == 0:
            raise ValueError(f"No supported image files were found under {self.root}")

        if dataset == "traffic_lights":
            required = {"red", "yellow", "green"}
            missing = required.difference(set(self.classes))
            if missing:
                raise ValueError(f"Traffic-light dataset is missing class folders: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return image, label

    def get_image(self, index: int) -> Image.Image:
        path, _ = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB").copy()


class TorchvisionGTSRBAdapter(Dataset):
    """Small adapter around torchvision.datasets.GTSRB, keeping PIL images untransformed."""

    def __init__(self, raw_dataset):
        self.raw = raw_dataset
        samples = getattr(raw_dataset, "_samples", None)
        if samples is not None:
            self.paths = [str(sample[0]) for sample in samples]
            self.targets = [int(sample[1]) for sample in samples]
        else:
            # Fallback for torchvision versions with a different internal representation.
            self.targets = []
            self.paths = []
            for i in range(len(raw_dataset)):
                _image, label = raw_dataset[i]
                self.targets.append(int(label))
                self.paths.append("")
        self.classes = GTSRB_CLASS_NAMES.copy()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image, label = self.raw[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return image.convert("RGB"), int(label)

    def get_image(self, index: int) -> Image.Image:
        image, _label = self[index]
        return image.copy()


def stratified_split(
    indices: Sequence[int], labels: Sequence[int], val_ratio: float = 0.15,
    test_ratio: float = 0.15, seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Class-aware split. For official GTSRB test data, pass only train/val ratios separately."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx in indices:
        by_class.setdefault(int(labels[idx]), []).append(int(idx))

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    for label, class_indices in sorted(by_class.items()):
        rng.shuffle(class_indices)
        n = len(class_indices)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
        train_ids.extend(class_indices[:n_train])
        val_ids.extend(class_indices[n_train:n_train + n_val])
        test_ids.extend(class_indices[n_train + n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    if not train_ids or not val_ids or not test_ids:
        raise ValueError(
            "The dataset is too small to create non-empty train/validation/test splits. "
            "Add more images per class (ideally at least 5)."
        )
    return train_ids, val_ids, test_ids


def split_train_validation(indices: Sequence[int], labels: Sequence[int], val_ratio: float = 0.15,
                           seed: int = 42) -> tuple[list[int], list[int]]:
    """Class-aware train/validation split when a separate official test set is available."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx in indices:
        by_class.setdefault(int(labels[idx]), []).append(int(idx))
    train_ids: list[int] = []
    val_ids: list[int] = []
    for _label, class_indices in sorted(by_class.items()):
        rng.shuffle(class_indices)
        n = len(class_indices)
        if n < 2:
            train_ids.extend(class_indices)
            continue
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)
        val_ids.extend(class_indices[:n_val])
        train_ids.extend(class_indices[n_val:])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    if not train_ids or not val_ids:
        raise ValueError("Not enough examples to make train/validation splits")
    return train_ids, val_ids


class TransformView(Dataset):
    """A view over a base dataset with its own transform and original sample indices."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int], transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        original_index = self.indices[item]
        image, label = self.base_dataset[original_index]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label), int(original_index)


def limit_per_class(indices: Sequence[int], labels: Sequence[int], max_per_class: int, seed: int) -> list[int]:
    if max_per_class <= 0:
        return list(indices)
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx in indices:
        by_class.setdefault(int(labels[idx]), []).append(int(idx))
    selected: list[int] = []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        selected.extend(class_indices[:max_per_class])
    rng.shuffle(selected)
    return selected
