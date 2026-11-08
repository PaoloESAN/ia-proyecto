"""Usar el modelo de clasificación de basura entrenado en cualquier máquina.

Soporta tanto el nuevo modelo de máxima precisión BasuraFix (96.34% en test con TTA,
MobileNetV3-Large sobre 8 clases unificadas de TrashNet + RealWaste + Garbage12)
como checkpoints antiguos de ResNet18 (`best_trashnet_notebook.pt`).

Uso por consola:
  py usar_modelo.py samples/botella_plastico.jpg
  py usar_modelo.py samples/botella_plastico.jpg --explain reports/gradcam.png
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from basurafix.config import DEFAULT_MODEL_PATH
from basurafix.dataset import load_image
from basurafix.predict import BasuraPredictor, main as predict_cli


def cargar_modelo(ckpt_path: str | Path = DEFAULT_MODEL_PATH):
    """Carga BasuraPredictor (por defecto) o un checkpoint legacy de ResNet18."""
    path = Path(ckpt_path)
    if not path.exists() and Path(DEFAULT_MODEL_PATH).exists():
        path = Path(DEFAULT_MODEL_PATH)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Formato BasuraFix (v1)
    if "arch" in ckpt or "classes" in ckpt:
        predictor = BasuraPredictor(path)
        clases_es = [predictor.class_names_es.get(c, c) for c in predictor.classes]
        return predictor, clases_es

    # Formato legacy (best_trashnet_notebook.pt)
    clases = ckpt["clases"]
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(clases))
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, clases


def predecir_probs(modelo, clases, img, tta: bool = True):
    """Devuelve (clase_ganadora, lista de probabilidades). Acepta PIL o ruta."""
    if isinstance(img, (str, Path)):
        img = load_image(img)
    else:
        img = img.convert("RGB")

    if isinstance(modelo, BasuraPredictor):
        res = modelo.predict(img, topk=len(modelo.classes), tta=tta)
        probs = [res["probabilities"][c] for c in modelo.classes]
        return res["label_es"], probs

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with torch.no_grad():
        tensor = tf(img).unsqueeze(0)
        probs = torch.softmax(modelo(tensor), 1)
        if tta:
            probs = (probs + torch.softmax(modelo(torch.flip(tensor, dims=[3])), 1)) / 2.0
        probs = probs[0]
    return clases[int(probs.argmax())], [float(p) for p in probs]


def predecir(modelo, clases, ruta_img):
    if isinstance(modelo, BasuraPredictor):
        res = modelo.predict_path(ruta_img, topk=3, tta=True)
        for item in res["top"]:
            print(f"  {item['label_es']:16s} {item['percent']:6.2f}%")
        print(f"-> {res['label_es']} ({res['container']})")
        if res.get("advice"):
            print(f"   Consejo: {res['advice']}")
        return

    clase, probs = predecir_probs(modelo, clases, ruta_img)
    for c, p in sorted(zip(clases, probs), key=lambda t: -t[1]):
        print(f"  {c:10s} {p:.2%}")
    print("->", clase)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: py usar_modelo.py <foto.jpg> [--explain salida.png]")
        sys.exit(1)
    sys.exit(predict_cli(sys.argv[1:]))
