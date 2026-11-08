"""Datasets, transformaciones y DataLoaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BASE_DIR, IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str | Path) -> Image.Image:
    """Abre una imagen, corrige orientación EXIF y la convierte a RGB."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (17, 28, 46))
            rgba = img.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            return background
        return img.convert("RGB")


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.60, 1.0), ratio=(0.80, 1.25), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12, expand=False),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.08
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"
                ),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(round(img_size * 1.14)), antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TrashDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        transform=None,
        base_dir: Path = BASE_DIR,
    ):
        frame = pd.read_csv(manifest)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No hay imágenes para el split '{split}' en {manifest}.")
        self.frame = frame
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.base_dir / path
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_id"]), str(path)

    @property
    def labels(self) -> list[int]:
        return self.frame["label_id"].astype(int).tolist()


def build_dataloaders(
    manifest: str | Path,
    img_size: int,
    batch_size: int,
    workers: int,
    classes: list[str],
) -> tuple[dict[str, DataLoader], dict[str, TrashDataset], dict[str, int]]:
    manifest = Path(manifest)
    frame = pd.read_csv(manifest)
    missing = sorted(set(classes) - set(frame["label"].unique()))
    if missing:
        raise ValueError(f"El manifest no contiene estas clases: {missing}")

    # Añadir label_id de forma determinista
    label_to_id = {name: index for index, name in enumerate(classes)}
    frame["label_id"] = frame["label"].map(label_to_id)
    frame.to_csv(manifest, index=False)

    datasets: dict[str, TrashDataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        if subset.empty:
            continue
        dataset = TrashDataset(manifest, split, build_transforms(img_size, split == "train"))
        datasets[split] = dataset
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=False,
        )
        if workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
        loaders[split] = DataLoader(dataset, **loader_kwargs)

    counts = frame["label"].value_counts().to_dict()
    return loaders, datasets, counts
