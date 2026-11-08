"""Descarga y verificación de los datasets de BasuraFix.

Fuentes:
  * TrashNet (Gary Thung & Mindy Yang, Stanford) - 6 clases, imágenes de estudio.
  * RealWaste (University of Wollongong) - 9 clases, imágenes reales de vertedero.
  * Garbage Classification (12 clases) - vidrio de colores, textil, orgánico y más.

Uso:
    python -m basurafix download
    python -m basurafix download --no-realwaste
    python -m basurafix download --no-garbage12
    python -m basurafix download --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .config import (
    GARBAGE12_DIR,
    GARBAGE12_EXTRACT,
    GARBAGE12_ZIP,
    RAW_DIR,
    REALWASTE_MAP,
    TRASHNET_CLASSES,
)
from .utils import format_duration, read_json, setup_console, write_json

TRASHNET_URL = (
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
)
TRASHNET_ZIP = RAW_DIR / "trashnet" / "dataset-resized.zip"
TRASHNET_DIR = RAW_DIR / "trashnet" / "dataset-resized"
REALWASTE_DIR = RAW_DIR / "realwaste"
REALWASTE_REPO = "shahzaibvohra/realwaste"
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
REALWASTE_ZIP = RAW_DIR / "realwaste" / "realwaste-uci.zip"

GARBAGE12_URL = (
    "https://huggingface.co/datasets/UdaraChamidu/"
    "Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip"
)


def _download(url: str, dest: Path, label: str, retries: int = 3) -> None:
    """Descarga con progreso y reintentos."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": "BasuraFix/1.0 (dataset downloader)"}
            with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                with open(tmp, "wb") as fh, tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    ncols=88,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc
            wait = 2 * attempt
            print(f"  Reintento {attempt}/{retries - 1} en {wait}s ({exc})")
            time.sleep(wait)


def download_trashnet(force: bool = False) -> int:
    """Descarga TrashNet y devuelve el número de imágenes."""
    if TRASHNET_DIR.exists() and not force:
        count = sum(1 for _ in TRASHNET_DIR.rglob("*.jpg"))
        if count > 2000:
            print(f"[TrashNet] Ya disponible: {count} imágenes en {TRASHNET_DIR}")
            return count
        shutil.rmtree(TRASHNET_DIR, ignore_errors=True)

    if force or not TRASHNET_ZIP.exists() or TRASHNET_ZIP.stat().st_size < 40_000_000:
        print("[TrashNet] Descargando dataset (~43 MB)...")
        _download(TRASHNET_URL, TRASHNET_ZIP, "TrashNet")

    print("[TrashNet] Extrayendo...")
    TRASHNET_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = TRASHNET_DIR.parent / "_extract_trashnet"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(TRASHNET_ZIP) as zf:
        zf.extractall(tmp_dir)

    extracted = tmp_dir / "dataset-resized"
    if not extracted.exists():
        candidates = [p for p in tmp_dir.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            extracted = candidates[0]
        else:
            raise RuntimeError("El ZIP de TrashNet no tiene la estructura esperada.")

    if TRASHNET_DIR.exists():
        shutil.rmtree(TRASHNET_DIR)
    shutil.move(str(extracted), str(TRASHNET_DIR))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    count = sum(1 for _ in TRASHNET_DIR.rglob("*.jpg"))
    print(f"[TrashNet] Listo: {count} imágenes en {TRASHNET_DIR}")
    return count


def _extract_realwaste_zip() -> None:
    print("[RealWaste] Extrayendo ZIP de UCI...")
    tmp_dir = RAW_DIR / "_extract_realwaste"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(REALWASTE_ZIP) as zf:
        zf.extractall(tmp_dir)
    # La raíz puede ser "RealWaste" o similar; mover todas las imágenes a REALWASTE_DIR
    REALWASTE_DIR.mkdir(parents=True, exist_ok=True)
    for path in tmp_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target = REALWASTE_DIR / path.name
            if target.exists():
                target = REALWASTE_DIR / f"{path.parent.name}_{path.name}"
            shutil.copy2(path, target)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_realwaste(force: bool = False) -> int:
    """Descarga RealWaste desde HuggingFace; si falla, usa el ZIP de UCI."""
    if REALWASTE_DIR.exists() and not force:
        count = sum(
            1
            for p in REALWASTE_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if count > 4000:
            print(f"[RealWaste] Ya disponible: {count} imágenes en {REALWASTE_DIR}")
            return count
        shutil.rmtree(REALWASTE_DIR, ignore_errors=True)

    REALWASTE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        print("[RealWaste] Descargando desde HuggingFace (~700 MB)...")
        snapshot_download(
            repo_id=REALWASTE_REPO,
            repo_type="dataset",
            local_dir=str(REALWASTE_DIR),
            allow_patterns=["*.jpg", "*.jpeg", "*.png"],
            max_workers=16,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[RealWaste] HuggingFace falló ({exc}). Probando espejo de UCI...")
        if force or not REALWASTE_ZIP.exists():
            _download(REALWASTE_URL, REALWASTE_ZIP, "RealWaste (UCI)")
        _extract_realwaste_zip()

    count = sum(
        1
        for p in REALWASTE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if count < 4000:
        raise RuntimeError(
            f"RealWaste quedó incompleto ({count} imágenes). Vuelve a ejecutar la descarga."
        )
    print(f"[RealWaste] Listo: {count} imágenes en {REALWASTE_DIR}")
    return count


def _count_by_class_trashnet() -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in TRASHNET_CLASSES:
        folder = TRASHNET_DIR / cls
        counts[cls] = sum(1 for _ in folder.glob("*.jpg")) if folder.exists() else 0
    return counts


def _count_images(root: Path) -> int:
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def download_garbage12(force: bool = False) -> int:
    """Descarga el dataset de 12 clases (~251 MB) y lo extrae."""
    if GARBAGE12_EXTRACT.exists() and not force:
        count = _count_images(GARBAGE12_EXTRACT)
        if count > 4000:
            print(f"[Garbage12] Ya disponible: {count} imágenes en {GARBAGE12_EXTRACT}")
            return count
        shutil.rmtree(GARBAGE12_EXTRACT, ignore_errors=True)

    if force or not GARBAGE12_ZIP.exists() or GARBAGE12_ZIP.stat().st_size < 200_000_000:
        print("[Garbage12] Descargando dataset de 12 clases (~251 MB)...")
        _download(GARBAGE12_URL, GARBAGE12_ZIP, "Garbage12")

    print("[Garbage12] Extrayendo...")
    tmp_dir = GARBAGE12_DIR / "_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    with zipfile.ZipFile(GARBAGE12_ZIP) as zf:
        zf.extractall(tmp_dir)
    if GARBAGE12_EXTRACT.exists():
        shutil.rmtree(GARBAGE12_EXTRACT, ignore_errors=True)
    shutil.move(str(tmp_dir), str(GARBAGE12_EXTRACT))

    count = _count_images(GARBAGE12_EXTRACT)
    if count < 4000:
        raise RuntimeError(
            f"El dataset de 12 clases quedó incompleto ({count} imágenes)."
        )
    print(f"[Garbage12] Listo: {count} imágenes en {GARBAGE12_EXTRACT}")
    return count


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Descarga TrashNet y RealWaste.")
    parser.add_argument("--no-trashnet", action="store_true", help="Omitir TrashNet.")
    parser.add_argument("--no-realwaste", action="store_true", help="Omitir RealWaste.")
    parser.add_argument(
        "--no-garbage12", action="store_true", help="Omitir el dataset de 12 clases."
    )
    parser.add_argument("--force", action="store_true", help="Volver a descargar todo.")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    sources: dict[str, dict] = read_json(RAW_DIR / "sources.json", default={}) or {}

    if not args.no_trashnet:
        count = download_trashnet(force=args.force)
        sources["trashnet"] = {
            "name": "TrashNet",
            "images": count,
            "url": "https://github.com/garythung/trashnet",
            "license": "MIT",
            "description": "Imágenes de residuos sobre fondo blanco (6 clases).",
            "classes": _count_by_class_trashnet(),
        }

    if not args.no_realwaste:
        count = download_realwaste(force=args.force)
        sources["realwaste"] = {
            "name": "RealWaste",
            "images": count,
            "url": "https://huggingface.co/datasets/shahzaibvohra/realwaste",
            "license": "CC BY 4.0",
            "description": "Residuos reales fotografiados en un vertedero (9 clases).",
            "classes": sorted(REALWASTE_MAP),
        }

    if not args.no_garbage12:
        try:
            count = download_garbage12(force=args.force)
            sources["garbage12"] = {
                "name": "Garbage Classification (12 clases)",
                "images": count,
                "url": "https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes",
                "license": "No especificada por el autor (uso educativo)",
                "description": "Dataset de 12 clases con vidrio de varios colores, ropa, calzado y biológicos.",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[Garbage12] No se pudo descargar ({exc}). Se continúa sin este dataset.")

    write_json(RAW_DIR / "sources.json", sources)
    print(f"\nDescarga completada en {format_duration(time.perf_counter() - started)}.")
    print("Siguiente paso:  python -m basurafix prepare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
