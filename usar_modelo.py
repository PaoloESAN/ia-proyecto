"""Usar el modelo TrashNet entrenado en cualquier máquina.

1) Instala dependencias (una vez):
     py -m pip install torch torchvision pillow

2) Copia a esa máquina ESTE script + best_trashnet_notebook.pt

3) Predice una foto:
     py usar_modelo.py mi_foto.jpg
"""
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def cargar_modelo(ckpt_path="best_trashnet_notebook.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    clases = ckpt["clases"]
    modelo = models.resnet18(weights=None)  # arquitectura, sin descargar pesos
    modelo.fc = nn.Linear(modelo.fc.in_features, len(clases))
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, clases


def predecir_probs(modelo, clases, img):
    """Devuelve (clase_ganadora, lista de probabilidades). Acepta PIL o ruta."""
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(modelo(tf(img).unsqueeze(0)), 1)[0]
    return clases[int(probs.argmax())], [float(p) for p in probs]


def predecir(modelo, clases, ruta_img):
    clase, probs = predecir_probs(modelo, clases, ruta_img)
    for c, p in sorted(zip(clases, probs), key=lambda t: -t[1]):
        print(f"  {c:10s} {p:.2%}")
    print("->", clase)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: py usar_modelo.py <foto.jpg>")
        sys.exit(1)
    m, c = cargar_modelo()
    predecir(m, c, sys.argv[1])
