"""Preprocesamiento: transforms train/eval mejorados y visualización.

Combina el pipeline avanzado de BasuraFix (EXIF transpose, fondo para PNGs
transparentes, RandomResizedCrop con antialias, GaussianBlur, RandomErasing y
Resize+CenterCrop en evaluación) con las funciones de visualización del notebook.
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile, ImageOps
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224
BATCH = 32

MEDIA = [0.485, 0.456, 0.406]
DESV = [0.229, 0.224, 0.225]


def load_image(path: str | Path) -> Image.Image:
    """Abre una imagen, corrige orientación EXIF y maneja transparencia RGBA."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (17, 28, 46))
            rgba = img.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            return background
        return img.convert("RGB")


def get_transforms(img_size=IMG_SIZE):
    """train_tf (con aumento avanzado BasuraFix) + eval_tf (Resize 1.14x + CenterCrop)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.60, 1.0), ratio=(0.80, 1.25), antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12, expand=False),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.08
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"
        ),
        transforms.Normalize(MEDIA, DESV),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14)), antialias=True),
        transforms.CenterCrop(img_size),
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

    batch = next(iter(train_ld))
    x, y = batch[0], batch[1]
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
