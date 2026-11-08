"""Inferencia: predicción con TTA y explicación Grad-CAM.

Uso:
    python -m basurafix predict "C:\\ruta\\foto.jpg"
    python -m basurafix predict foto.jpg --explain reports/explicacion.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

try:  # Formatos modernos opcionales
    import pillow_avif  # noqa: F401
except Exception:
    pass
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

from .config import (
    CLASS_INFO,
    CLASS_NAMES_ES,
    CLASSES,
    DEFAULT_MODEL_PATH,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .dataset import load_image
from .gradcam import GradCAM, overlay_heatmap
from .model import load_checkpoint
from .utils import setup_console

UNCERTAIN_CONFIDENCE = 0.45
UNCERTAIN_MARGIN = 0.10


class BasuraPredictor:
    """Carga el checkpoint una vez y predice imágenes."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(model_path, self.device)
        self.classes: list[str] = self.checkpoint.get("classes", CLASSES)
        self.class_names_es: dict[str, str] = self.checkpoint.get(
            "class_names_es", CLASS_NAMES_ES
        )
        self.class_info: dict[str, dict] = self.checkpoint.get("class_info", CLASS_INFO)
        self.img_size = int(self.checkpoint.get("img_size", 224))
        mean = self.checkpoint.get("mean", IMAGENET_MEAN)
        std = self.checkpoint.get("std", IMAGENET_STD)
        self.transform = _eval_transform(self.img_size, mean, std)
        self._gradcam: GradCAM | None = None

    # ------------------------------------------------------------------
    def predict(
        self,
        image: Image.Image,
        topk: int = 3,
        tta: bool = True,
    ) -> dict:
        started = time.perf_counter()
        image = _ensure_rgb(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            if tta:
                flipped = torch.flip(tensor, dims=[3])
                probabilities = (
                    probabilities + torch.softmax(self.model(flipped), dim=1)
                ) / 2.0
        probabilities = probabilities[0].cpu()

        values, indices = torch.topk(probabilities, k=min(topk, len(self.classes)))
        top = []
        for value, index in zip(values.tolist(), indices.tolist()):
            label = self.classes[index]
            info = self.class_info.get(label, {})
            top.append(
                {
                    "label": label,
                    "label_es": self.class_names_es.get(label, label),
                    "prob": float(value),
                    "percent": float(value) * 100.0,
                    "color": info.get("color", "#94a3b8"),
                    "container": info.get("container", ""),
                }
            )

        confidence = top[0]["prob"]
        margin = top[0]["prob"] - (top[1]["prob"] if len(top) > 1 else 0.0)
        best = top[0]
        warning = None
        if confidence < UNCERTAIN_CONFIDENCE:
            warning = "Confianza baja. Prueba con mejor luz, fondo limpio y el objeto centrado."
        elif margin < UNCERTAIN_MARGIN:
            warning = "El resultado es ambiguo: hay dos clases muy parecidas. Revisa la foto."

        return {
            "label": best["label"],
            "label_es": best["label_es"],
            "confidence": confidence,
            "confidence_percent": best["percent"],
            "margin": margin,
            "top": top,
            "probabilities": {
                label: float(prob)
                for label, prob in zip(self.classes, probabilities.tolist())
            },
            "container": best["container"],
            "advice": self.class_info.get(best["label"], {}).get("tip", ""),
            "warning": warning,
            "uncertain": warning is not None,
            "tta": tta,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    def predict_path(self, path: str | Path, topk: int = 3, tta: bool = True) -> dict:
        return self.predict(load_image(path), topk=topk, tta=tta)

    # ------------------------------------------------------------------
    def explain(self, image: Image.Image, size: int = 512) -> dict:
        image = _ensure_rgb(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model, self.checkpoint.get("arch", ""))
        cam, class_index = self._gradcam.generate(tensor)
        original, heatmap = overlay_heatmap(image, cam, size=size)
        return {
            "original": original,
            "heatmap": heatmap,
            "class_index": class_index,
            "label": self.classes[class_index],
            "label_es": self.class_names_es.get(self.classes[class_index], ""),
        }

    def info(self) -> dict:
        metrics = self.checkpoint.get("metrics", {})
        dataset = self.checkpoint.get("dataset", {})
        return {
            "arch": self.checkpoint.get("arch", "desconocida"),
            "img_size": self.img_size,
            "classes": [self.class_names_es.get(c, c) for c in self.classes],
            "trained_at": self.checkpoint.get("trained_at", ""),
            "version": self.checkpoint.get("version", ""),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_accuracy_tta": metrics.get("test_accuracy_tta"),
            "val_accuracy": metrics.get("val_accuracy"),
            "total_images": dataset.get("total_images"),
            "splits": dataset.get("splits", {}),
            "sources": self.checkpoint.get("sources", {}),
        }


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (17, 28, 46))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def _eval_transform(img_size: int, mean, std):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(round(img_size * 1.14)), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _print_result(result: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  {result['label_es'].upper()}  ·  {result['confidence_percent']:.1f}% de confianza")
    print("=" * 62)
    for index, item in enumerate(result["top"], start=1):
        bar = "#" * int(round(item["percent"] / 2.5))
        print(f"  {index}. {item['label_es']:<16} {item['percent']:6.1f}%  {bar}")
    print(f"\n  {result['container']}")
    if result["advice"]:
        print(f"  {result['advice']}")
    if result["warning"]:
        print(f"\n  [!] {result['warning']}")
    print(f"\n  Inferencia: {result['elapsed_ms']:.0f} ms · TTA: {'sí' if result['tta'] else 'no'}")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Clasifica una imagen de basura.")
    parser.add_argument("image", type=Path, help="Ruta de la imagen.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--json", action="store_true", help="Salida JSON.")
    parser.add_argument("--explain", type=Path, help="Guarda el Grad-CAM en esta ruta.")
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"No existe la imagen: {args.image}", file=sys.stderr)
        return 1
    try:
        predictor = BasuraPredictor(args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo cargar el modelo: {exc}", file=sys.stderr)
        return 1

    try:
        image = load_image(args.image)
        result = predictor.predict(image, topk=args.topk, tta=not args.no_tta)
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo analizar la imagen: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_result(result)

    if args.explain:
        explanation = predictor.explain(image)
        args.explain.parent.mkdir(parents=True, exist_ok=True)
        comparison = Image.new("RGB", (explanation["original"].width * 2 + 12, explanation["original"].height), (15, 23, 42))
        comparison.paste(explanation["original"], (0, 0))
        comparison.paste(explanation["heatmap"], (explanation["original"].width + 12, 0))
        comparison.save(args.explain)
        print(f"\nGrad-CAM guardado en {args.explain}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
