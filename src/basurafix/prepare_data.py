"""Prepara un manifest unificado a partir de TrashNet, RealWaste y datasets extra.

- Mapea las etiquetas originales a 8 clases comunes.
- Elimina duplicados exactos (SHA-1).
- Genera un split estratificado train/val/test reproducible.
- No copia imágenes: guarda rutas relativas en data/processed/manifest.csv.

Uso:
    python -m basurafix prepare
    python -m basurafix prepare --extra "kaggle=C:\\ruta\\dataset"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (
    BASE_DIR,
    CLASS_INFO,
    CLASSES,
    GARBAGE12_EXTRACT,
    MANIFEST_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    REALWASTE_MAP,
    REPORTS_DIR,
    STATS_PATH,
    TRASHNET_CLASSES,
    map_label,
    normalize_label,
)
from .utils import format_duration, read_json, setup_console, write_json

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    source: str


def _images(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
            yield path


def _parse_flat_label(stem: str) -> str | None:
    """'Food Organics_123' -> 'Food Organics'."""
    base = re.sub(r"[\s_\-]*\d+$", "", stem).strip()
    if not base:
        base = stem
    return base


def collect_trashnet() -> list[Sample]:
    root = RAW_DIR / "trashnet" / "dataset-resized"
    samples: list[Sample] = []
    for cls in TRASHNET_CLASSES:
        folder = root / cls
        for path in _images(folder):
            samples.append(Sample(path=path, label=cls, source="trashnet"))
    return samples


def collect_realwaste() -> list[Sample]:
    root = RAW_DIR / "realwaste"
    samples: list[Sample] = []
    for path in _images(root):
        # 1) si viene en carpetas por clase
        parent = path.parent.name.strip().lower()
        label = REALWASTE_MAP.get(parent)
        # 2) si viene plano: "Miscellaneous Trash_123.jpg"
        if label is None:
            base = _parse_flat_label(path.stem)
            label = REALWASTE_MAP.get(base.strip().lower())
        if label is None:
            # 3) variantes con guiones bajos
            base = normalize_label(_parse_flat_label(path.stem)).replace("_", " ")
            label = REALWASTE_MAP.get(base)
        if label is not None:
            samples.append(Sample(path=path, label=label, source="realwaste"))
    return samples


def collect_extra(name: str, root: Path) -> list[Sample]:
    """Recolecta un dataset extra con carpetas por clase (o archivos planos).

    Acepta estructuras como ``root/plastic/*.jpg``, ``root/train/plastic/*.jpg``
    o ``root/mi_clase/*.jpg``. Las carpetas cuyo nombre no es una clase conocida
    (p. ej. ``train``, ``test``) se ignoran y se recorren sus subcarpetas.
    """
    samples: list[Sample] = []
    if not root.exists():
        print(f"  [aviso] No existe el dataset extra '{name}': {root}")
        return samples

    # Si la raíz misma es una carpeta de clase: --extra "plastic=C:\...\plastic"
    root_label = map_label(root.name)
    if root_label is not None:
        for path in _images(root):
            samples.append(Sample(path=path, label=root_label, source=name))
        return samples

    class_folders = [
        folder for folder in sorted(root.rglob("*")) if folder.is_dir() and map_label(folder.name)
    ]
    if class_folders:
        for folder in class_folders:
            label = map_label(folder.name)
            if label is None:
                continue
            for path in _images(folder):
                samples.append(Sample(path=path, label=label, source=name))
    else:
        # Dataset plano: "Plastic_001.jpg"
        for path in _images(root):
            label = map_label(_parse_flat_label(path.stem))
            if label is not None:
                samples.append(Sample(path=path, label=label, source=name))
    return samples


def collect_garbage12(max_per_class: int = 1200, seed: int = 42) -> list[Sample]:
    """Recolecta el dataset de 12 clases con un tope por clase unificada.

    Las clases 'battery' se ignoran (residuo peligroso, no es basura común).
    'clothes' + 'shoes' se unifican en textil y los 3 vidrios en vidrio.
    """
    if not GARBAGE12_EXTRACT.exists():
        return []
    samples: list[Sample] = []
    for path in _images(GARBAGE12_EXTRACT):
        label = map_label(path.parent.name)
        if label is None:
            label = map_label(_parse_flat_label(path.stem))
        if label is not None:
            samples.append(Sample(path=path, label=label, source="garbage12"))

    if max_per_class <= 0:
        return samples

    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample)
    rng = np.random.default_rng(seed)
    capped: list[Sample] = []
    for label in CLASSES:
        items = by_class.get(label, [])
        if len(items) > max_per_class:
            indices = rng.choice(len(items), size=max_per_class, replace=False)
            items = [items[int(i)] for i in sorted(indices)]
        capped.extend(items)
    return capped


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(samples: list[Sample]) -> tuple[list[Sample], list[dict], dict[str, str]]:
    seen: dict[str, Sample] = {}
    unique: list[Sample] = []
    duplicates: list[dict] = []
    hashes: dict[str, str] = {}
    for sample in samples:
        try:
            digest = _sha1(sample.path)
        except OSError as exc:
            print(f"  [aviso] No se pudo leer {sample.path}: {exc}")
            continue
        if digest in seen:
            duplicates.append(
                {
                    "kept": str(seen[digest].path.relative_to(BASE_DIR)),
                    "removed": str(sample.path.relative_to(BASE_DIR)),
                }
            )
            continue
        seen[digest] = sample
        unique.append(sample)
        hashes[str(sample.path)] = digest
    return unique, duplicates, hashes


def _stratified_split(
    samples: list[Sample], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample)

    assignment: dict[str, str] = {}
    for label in CLASSES:
        items = by_class.get(label, [])
        if not items:
            continue
        order = rng.permutation(len(items))
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 10:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        else:
            n_test = 0
            n_val = 0
        for position, index in enumerate(order):
            if position < n_test:
                split = "test"
            elif position < n_test + n_val:
                split = "val"
            else:
                split = "train"
            assignment[str(items[index].path)] = split
    return assignment


def _plot_distribution(counts: dict[str, dict[str, int]], sources: dict[str, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [CLASS_INFO[c]["es"] for c in CLASSES]
    train = [counts[c]["train"] for c in CLASSES]
    val = [counts[c]["val"] for c in CLASSES]
    test = [counts[c]["test"] for c in CLASSES]
    colors = [CLASS_INFO[c]["color"] for c in CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, train, color=colors, alpha=0.95, label="Train")
    ax.barh(y, val, left=train, color=colors, alpha=0.62, label="Validación")
    ax.barh(y, test, left=np.array(train) + np.array(val), color=colors, alpha=0.32, label="Test")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Imágenes")
    ax.set_title("Distribución por clase y split", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    for i, total in enumerate(np.array(train) + np.array(val) + np.array(test)):
        ax.text(total + 8, i, str(total), va="center", fontsize=9, color="#334155")

    ax2 = axes[1]
    src_labels = list(sources.keys())
    src_values = [sources[k] for k in src_labels]
    palette = ["#22c55e", "#38bdf8", "#f59e0b", "#a78bfa", "#ef4444"]
    ax2.bar(src_labels, src_values, color=palette[: len(src_labels)])
    ax2.set_title("Imágenes por fuente", fontweight="bold")
    ax2.set_ylabel("Imágenes")
    for i, value in enumerate(src_values):
        ax2.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = REPORTS_DIR / "dataset_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado en {out}")


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Unifica y divide los datasets.")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NOMBRE=RUTA",
        help="Dataset adicional con carpetas por clase. Se puede repetir.",
    )
    parser.add_argument("--val", type=float, default=0.15, help="Fracción de validación.")
    parser.add_argument("--test", type=float, default=0.15, help="Fracción de test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--garbage12-max",
        type=int,
        default=1200,
        help="Máximo de imágenes por clase del dataset de 12 clases (0 = sin límite).",
    )
    parser.add_argument(
        "--no-garbage12",
        action="store_true",
        help="No usar el dataset de 12 clases aunque esté descargado.",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Borra el manifest anterior antes de generar."
    )
    args = parser.parse_args(argv)

    if args.val <= 0 or args.test <= 0 or args.val + args.test >= 1:
        parser.error("Las fracciones val/test deben ser > 0 y sumar menos de 1.")

    started = time.perf_counter()
    print("Recolectando imágenes...")
    samples: list[Sample] = []
    n_trashnet = len(collect_trashnet())
    n_realwaste = len(collect_realwaste())
    samples.extend(collect_trashnet())
    samples.extend(collect_realwaste())
    print(f"  TrashNet:  {n_trashnet:,} imágenes")
    print(f"  RealWaste: {n_realwaste:,} imágenes")

    if not args.no_garbage12:
        garbage12 = collect_garbage12(args.garbage12_max, args.seed)
        if garbage12:
            print(
                f"  Garbage12: {len(garbage12):,} imágenes "
                f"(tope {args.garbage12_max or 'sin límite'} por clase)"
            )
            samples.extend(garbage12)

    for spec in args.extra:
        if "=" not in spec:
            parser.error(f"Formato inválido para --extra: {spec!r} (usa NOMBRE=RUTA)")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        extra = collect_extra(name.strip(), path)
        samples.extend(extra)
        print(f"  {name}: {len(extra):,} imágenes")

    if not samples:
        print(
            "\nNo se encontraron imágenes. Ejecuta primero:\n"
            "  python -m basurafix download",
            file=sys.stderr,
        )
        return 1

    # Quedarse solo con clases válidas
    samples = [s for s in samples if s.label in CLASSES]

    print("\nEliminando duplicados exactos...")
    unique, duplicates, hashes = _dedupe(samples)
    print(f"  {len(unique):,} imágenes únicas ({len(duplicates)} duplicados eliminados)")

    print("Generando split estratificado...")
    assignment = _stratified_split(unique, args.val, args.test, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    tmp_manifest = MANIFEST_PATH.with_suffix(".csv.part")
    with open(tmp_manifest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "source", "split", "sha1"])
        for sample in unique:
            rel = str(sample.path.relative_to(BASE_DIR)).replace("\\", "/")
            writer.writerow(
                [
                    rel,
                    sample.label,
                    sample.source,
                    assignment[str(sample.path)],
                    hashes.get(str(sample.path), ""),
                ]
            )
    tmp_manifest.replace(MANIFEST_PATH)

    # Estadísticas
    counts: dict[str, dict[str, int]] = {
        cls: {"train": 0, "val": 0, "test": 0, "total": 0} for cls in CLASSES
    }
    source_counts: Counter[str] = Counter()
    source_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in unique:
        split = assignment[str(sample.path)]
        counts[sample.label][split] += 1
        counts[sample.label]["total"] += 1
        source_counts[sample.source] += 1
        source_by_class[sample.source][sample.label] += 1

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_images": len(unique),
        "duplicates_removed": len(duplicates),
        "classes": counts,
        "sources": {
            name: {"total": source_counts[name], "classes": dict(source_by_class[name])}
            for name in source_counts
        },
        "split": {"val_fraction": args.val, "test_fraction": args.test, "seed": args.seed},
    }
    write_json(STATS_PATH, stats)

    # Distribución
    source_totals = {name: source_counts[name] for name in source_counts}
    _plot_distribution(counts, source_totals)

    # Tabla resumen
    print("\n" + "=" * 58)
    print(f"{'Clase':22} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 58)
    for cls in CLASSES:
        c = counts[cls]
        print(
            f"{CLASS_INFO[cls]['es']:22} {c['total']:>8,} {c['train']:>8,} "
            f"{c['val']:>8,} {c['test']:>8,}"
        )
    print("-" * 58)
    totals = {k: sum(counts[c][k] for c in CLASSES) for k in ("total", "train", "val", "test")}
    print(
        f"{'TOTAL':22} {totals['total']:>8,} {totals['train']:>8,} "
        f"{totals['val']:>8,} {totals['test']:>8,}"
    )
    print("=" * 58)
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Preparación completada en {format_duration(time.perf_counter() - started)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
