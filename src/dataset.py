"""Lectura del dataset TrashNet (espejo HuggingFace) y split estratificado train/val/test.

También expone utilidades del pipeline multi-dataset de BasuraFix
(TrashNet + RealWaste + Garbage12, 8 clases unificadas, deduplicación SHA-1).
"""
from collections import defaultdict
import random

import numpy as np
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


def split_indices(n_total, seed=SEED, p_train=0.70, p_val=0.13, labels=None):
    """Divide indices en train/val/test (estratificado si se provee `labels`)."""
    if labels is not None:
        rng = np.random.default_rng(seed)
        by_class = defaultdict(list)
        for idx, lbl in enumerate(labels):
            by_class[int(lbl)].append(idx)
        train_idx, val_idx, test_idx = [], [], []
        p_test = max(0.0, 1.0 - p_train - p_val)
        for items in by_class.values():
            order = rng.permutation(len(items))
            n = len(items)
            n_test = int(round(n * p_test))
            n_val = int(round(n * p_val))
            for pos, k in enumerate(order):
                if pos < n_test:
                    test_idx.append(items[k])
                elif pos < n_test + n_val:
                    val_idx.append(items[k])
                else:
                    train_idx.append(items[k])
        return train_idx, val_idx, test_idx

    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    n_tr = int(p_train * n_total)
    n_va = int(p_val * n_total)
    return indices[:n_tr], indices[n_tr:n_tr + n_va], indices[n_tr + n_va:]
