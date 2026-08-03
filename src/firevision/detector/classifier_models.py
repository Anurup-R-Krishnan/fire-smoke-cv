from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True, slots=True)
class ClassifierBundle:
    model: nn.Module
    target_layer: nn.Module


def build_classifier(name: str, class_count: int, pretrained: bool) -> ClassifierBundle:
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return ClassifierBundle(model=model, target_layer=model.features[-1])
    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, class_count)
        return ClassifierBundle(model=model, target_layer=model.features[-1])
    raise ValueError(f"Unsupported classifier architecture: {name}")


def freeze_backbone(name: str, model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    classifier = getattr(model, "classifier")
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_feature_blocks(name: str, model: nn.Module, block_count: int) -> None:
    if block_count <= 0:
        return
    features = getattr(model, "features")
    children = list(features.children())
    for block in children[-block_count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in getattr(model, "classifier").parameters():
        parameter.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}
