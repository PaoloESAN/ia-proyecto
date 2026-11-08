"""Grad-CAM para explicar las predicciones del modelo."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, arch: str):
        self.model = model
        self.arch = arch
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = []
        layer = self._target_layer(model, arch)
        self._handles.append(layer.register_forward_hook(self._forward_hook))
        self._handles.append(layer.register_full_backward_hook(self._backward_hook))

    @staticmethod
    def _target_layer(model: torch.nn.Module, arch: str):
        if arch.startswith(("mobilenet", "efficientnet", "convnext")):
            return model.features[-1]
        if arch.startswith("resnet"):
            return model.layer4[-1]
        raise ValueError(f"Grad-CAM no soportado para {arch}")

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(self, tensor: torch.Tensor, class_index: int | None = None):
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        logits = self.model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("No se pudieron capturar activaciones para Grad-CAM.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(
            cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_index


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45, size: int = 512
) -> tuple[Image.Image, Image.Image]:
    """Devuelve (imagen original redimensionada, imagen con el mapa de calor)."""
    import matplotlib

    base = image.convert("RGB")
    width, height = base.size
    scale = size / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    base = base.resize(new_size, Image.Resampling.LANCZOS)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        new_size, Image.Resampling.BILINEAR
    )
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0
    cmap = matplotlib.colormaps["turbo"]
    heat = (cmap(cam_arr)[..., :3] * 255).astype(np.uint8)
    original = np.asarray(base).astype(np.float32)
    blended = (original * (1 - alpha) + heat * alpha).astype(np.uint8)
    return base, Image.fromarray(blended)
