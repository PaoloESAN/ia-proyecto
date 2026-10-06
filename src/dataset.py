"""Lectura del dataset TrashNet (espejo HuggingFace) y split train/val/test.

Basado en Pasos 2 y 4 de TrashNet_Clasificacion.ipynb.
"""
import random

from torch.utils.data import Dataset


DATASET_ID = "kuchidareo/small_trashnet"
SEED = 42


def load_trashnet():
    """Descarga el dataset y devuelve (hf_dataset, clases)."""
    from datasets import load_dataset

    hf = load_dataset(DATASET_ID, split="train")
    clases = hf.features["label"].names
    return hf, clases


def contar_por_clase(hf, clases):
    """Devuelve dict {clase: n_imagenes}."""
    conteo = {c: 0 for c in clases}
    for i in range(len(hf)):
        conteo[clases[int(hf[i]["label"])]] += 1
    return conteo


class TrashDataset(Dataset):
    """Envuelve el dataset de HuggingFace como (imagen, etiqueta)."""

    def __init__(self, hf_ds, indices, transform):
        self.hf, self.idx, self.tf = hf_ds, indices, transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.hf[self.idx[i]]
        return self.tf(r["image"].convert("RGB")), int(r["label"])


def split_indices(n_total, seed=SEED, p_train=0.70, p_val=0.13):
    """Divide indices en train/val/test (70/13/17 como el paper original)."""
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_tr = int(p_train * n_total)
    n_va = int(p_val * n_total)
    return indices[:n_tr], indices[n_tr:n_tr + n_va], indices[n_tr + n_va:]
