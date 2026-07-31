from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .classifier_data import build_transforms
from .classifier_models import build_classifier
from .config import DetectorTrainingConfig
from .runtime import resolve_device


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._capture_tensor_gradient)

    def _capture_tensor_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient.detach()

    def close(self) -> None:
        self.forward_handle.remove()

    def generate(self, tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        predicted = int(logits.argmax(dim=1).item())
        selected = predicted if target_class is None else int(target_class)
        logits[:, selected].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap[0]
        minimum = heatmap.min()
        maximum = heatmap.max()
        heatmap = (heatmap - minimum) / (maximum - minimum + 1e-8)
        return heatmap.detach().cpu().numpy(), predicted


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    colour = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.58, colour, 0.42, 0)


def load_classifier_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, tuple[str, ...], int, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = str(checkpoint["architecture"])
    class_names = tuple(str(value) for value in checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])
    bundle = build_classifier(architecture, len(class_names), pretrained=False)
    bundle.model.load_state_dict(checkpoint["state_dict"])
    bundle.model.to(device).eval()
    return bundle.model, bundle.target_layer, class_names, image_size, architecture


def generate_failure_gallery(
    config: DetectorTrainingConfig,
    checkpoint_path: Path,
    predictions_csv: Path,
    maximum_images: int = 16,
) -> list[Path]:
    device = resolve_device(config.device)
    model, target_layer, class_names, image_size, architecture = load_classifier_checkpoint(
        checkpoint_path, device
    )
    _, evaluation_transform = build_transforms(image_size)
    rows = list(csv.DictReader(predictions_csv.open(encoding="utf-8")))
    failures = [row for row in rows if row.get("correct", "").lower() in {"false", "0"}]
    selected = failures[:maximum_images]
    output_dir = config.output.report_dir / architecture / "gradcam_failures"
    output_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model, target_layer)
    outputs: list[Path] = []
    try:
        for index, row in enumerate(selected):
            path = Path(row["path"])
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            with Image.open(path) as image:
                tensor = evaluation_transform(image.convert("RGB")).unsqueeze(0).to(device)
            true_class = class_names.index(row["true_class"])
            heatmap, predicted = cam.generate(tensor, target_class=predicted_from_row(row, class_names))
            overlay = overlay_heatmap(image_bgr, heatmap)
            label = f"true={class_names[true_class]} predicted={class_names[predicted]}"
            cv2.putText(
                overlay,
                label,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            destination = output_dir / f"failure_{index:03d}.jpg"
            cv2.imwrite(str(destination), overlay)
            outputs.append(destination)
    finally:
        cam.close()
    return outputs


def predicted_from_row(row: dict[str, str], class_names: tuple[str, ...]) -> int:
    name = row.get("predicted_class", "")
    return class_names.index(name) if name in class_names else 0
