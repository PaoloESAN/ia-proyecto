"""Preprocesamiento: transforms train/eval y visualización.

Basado en Pasos 3 y 4 de TrashNet_Clasificacion.ipynb.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

IMG_SIZE = 224
BATCH = 32

MEDIA = [0.485, 0.456, 0.406]
DESV = [0.229, 0.224, 0.225]


def get_transforms(img_size=IMG_SIZE):
    """train_tf (con aumento) + eval_tf (determinista)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESV),
    ])
    return train_tf, eval_tf


def mostrar_ejemplos(hf, clases, n=12, seed=None):
    """Grid 3x4 con fotos aleatorias del dataset crudo."""
    if seed is not None:
        random.seed(seed)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for ax in axes.flat:
        i = random.randint(0, len(hf) - 1)
        ax.imshow(hf[i]["image"].convert("RGB"))
        ax.set_title(clases[int(hf[i]["label"])])
        ax.axis("off")
    plt.suptitle("Ejemplos del dataset TrashNet")
    plt.tight_layout()
    plt.show()


def mostrar_lote(train_ld, clases):
    """Grid 2x4 con un lote ya aumentado (desnormalizado para ver)."""
    from src.dataset import TrashDataset  # noqa: F401 (evita import circular en docs)

    x, y = next(iter(train_ld))
    print("Lote:", tuple(x.shape))
    media = np.array(MEDIA)
    desv = np.array(DESV)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for k, ax in enumerate(axes.flat):
        foto = (x[k].numpy().transpose(1, 2, 0) * desv + media).clip(0, 1)
        ax.imshow(foto)
        ax.set_title(clases[int(y[k])])
        ax.axis("off")
    plt.suptitle("Un lote de entrenamiento (con aumento de datos)")
    plt.tight_layout()
    plt.show()
