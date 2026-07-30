from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DetectorTrainingConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class PatchSample:
    path: Path
    class_id: int
    class_name: str


class PatchFolderDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        root: Path,
        split: str,
        class_names: tuple[str, ...],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.root = root
        self.split = split
        self.class_names = class_names
        self.transform = transform
        self.samples: list[PatchSample] = []
        for class_id, class_name in enumerate(class_names):
            class_dir = root / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing Classical ML patch directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append(PatchSample(path, class_id, class_name))
        if not self.samples:
            raise ValueError(f"No images found in {root / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, sample.class_id, str(sample.path)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    resize_size = int(round(image_size * 1.12))
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.12, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.12),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return train_transform, evaluation_transform


def build_datasets(config: DetectorTrainingConfig) -> dict[str, PatchFolderDataset]:
    train_transform, evaluation_transform = build_transforms(config.classifier.image_size)
    return {
        "train": PatchFolderDataset(
            config.classification_patch_dir,
            "train",
            config.classifier_classes,
            train_transform,
        ),
        "val": PatchFolderDataset(
            config.classification_patch_dir,
            "val",
            config.classifier_classes,
            evaluation_transform,
        ),
        "test": PatchFolderDataset(
            config.classification_patch_dir,
            "test",
            config.classifier_classes,
            evaluation_transform,
        ),
    }


def build_loaders(
    config: DetectorTrainingConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, PatchFolderDataset], dict[str, DataLoader]]:
    datasets = build_datasets(config)
    pin_memory = device.type == "cuda"
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=config.classifier.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.classifier.num_workers > 0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
    return datasets, loaders


def balanced_class_weights(dataset: PatchFolderDataset, class_count: int) -> torch.Tensor:
    counts = torch.zeros(class_count, dtype=torch.float32)
    for sample in dataset.samples:
        counts[sample.class_id] += 1
    if torch.any(counts == 0):
        missing = [dataset.class_names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training dataset has no samples for: {missing}")
    total = counts.sum()
    return total / (class_count * counts)
