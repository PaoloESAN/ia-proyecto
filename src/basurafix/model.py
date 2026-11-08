"""Arquitecturas CNN con transfer learning y utilidades de checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASS_INFO, CLASS_NAMES_ES, CLASSES


def build_model(
    arch: str = "mobilenet_v3_large",
    num_classes: int = len(CLASSES),
    pretrained: bool = True,
    dropout: float = 0.25,
) -> nn.Module:
    """Crea una CNN preentrenada en ImageNet con la cabeza adaptada."""
    arch = arch.lower()
    weights = "DEFAULT" if pretrained else None

    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout)
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout)
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Congela todo salvo la cabeza clasificadora."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("classifier", "fc"))


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def frozen_modules_eval(model: nn.Module) -> None:
    """Pone en modo eval los módulos congelados (BatchNorm no actualiza stats)."""
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            module.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, head_lr_multiplier: float = 1.0
) -> torch.optim.Optimizer:
    """LR más bajo para el backbone preentrenado y más alto para la cabeza."""
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head if name.startswith(("classifier", "fc")) else backbone).append(parameter)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    if head:
        groups.append({"params": head, "lr": lr * head_lr_multiplier})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Carga un checkpoint de BasuraFix y reconstruye el modelo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo {path}. Entrena primero con: python -m basurafix train"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", CLASSES)
    arch = checkpoint.get("arch", "mobilenet_v3_large")
    model = build_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def checkpoint_payload(
    model: nn.Module,
    arch: str,
    img_size: int,
    metrics: dict,
    history: list[dict],
    sources: dict,
    dataset: dict,
    extra: dict | None = None,
) -> dict:
    payload = {
        "format_version": 1,
        "arch": arch,
        "classes": CLASSES,
        "class_names_es": CLASS_NAMES_ES,
        "class_info": CLASS_INFO,
        "img_size": img_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "history": history,
        "sources": sources,
        "dataset": dataset,
    }
    if extra:
        payload.update(extra)
    return payload
