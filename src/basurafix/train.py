"""Entrenamiento de la CNN con transfer learning en dos fases.

Fase 1: backbone congelado, se entrena solo la cabeza.
Fase 2: fine-tuning completo con LR bajo, cosine schedule y early stopping.

Uso:
    python -m basurafix train
    python -m basurafix train --arch efficientnet_b0 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from .config import (
    CLASS_INFO,
    CLASS_NAMES_ES,
    CLASSES,
    DEFAULT_ARCH,
    DEFAULT_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    STATS_PATH,
)
from .dataset import build_dataloaders
from .model import (
    build_model,
    build_optimizer,
    checkpoint_payload,
    count_trainable,
    freeze_backbone,
    frozen_modules_eval,
    unfreeze_all,
)
from .utils import format_duration, read_json, seed_everything, setup_console, write_json


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _class_weights(labels: list[int], num_classes: int = len(CLASSES)) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (num_classes * counts)) ** 0.5
    weights = np.clip(weights, 0.35, 3.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    phase1: bool = False,
    amp: bool = False,
) -> dict:
    training = optimizer is not None
    model.train(training)
    if training and phase1:
        frozen_modules_eval(model)

    total_loss = 0.0
    total = 0
    correct = 0
    top3_correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.detach()) * targets.size(0)
            total += targets.size(0)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum())
            top3_correct += int(
                (logits.topk(min(3, logits.shape[1]), dim=1).indices == targets.unsqueeze(1))
                .any(dim=1)
                .sum()
            )
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    accuracy = correct / max(total, 1)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": accuracy,
        "top3_accuracy": top3_correct / max(total, 1),
        "macro_f1": macro_f1,
    }


@torch.inference_mode()
def _evaluate_tta(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Precisión/F1 promediando la imagen original y su espejo horizontal."""
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    for images, targets, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        probs += torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        predictions = probs.argmax(dim=1).cpu()
        correct += int((predictions == targets).sum())
        total += targets.size(0)
        y_true.extend(targets.tolist())
        y_pred.extend(predictions.tolist())
    return correct / max(total, 1), float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def _save_history_plot(history: list[dict], phase_boundary: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train", color="#38bdf8", lw=2)
    ax.plot(epochs, [h["val_loss"] for h in history], label="Validación", color="#f59e0b", lw=2)
    ax.set_title("Pérdida", fontweight="bold")
    ax.set_xlabel("Época")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(epochs, [h["train_f1"] for h in history], label="Train", color="#22c55e", lw=2)
    ax.plot(epochs, [h["val_f1"] for h in history], label="Validación", color="#a78bfa", lw=2)
    ax.set_title("F1 macro", fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    for axis in axes:
        axis.axvline(phase_boundary + 0.5, color="#94a3b8", ls="--", lw=1)
    axes[1].text(
        phase_boundary + 0.7,
        0.04,
        "fine-tuning →",
        color="#64748b",
        fontsize=8,
    )
    fig.suptitle("BasuraFix · curva de entrenamiento", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confusion(y_true: list[int], y_pred: list[int], out: Path) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    labels = [CLASS_INFO[c]["es"] for c in CLASSES]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (test, normalizada)", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = norm[i, j]
            color = "white" if value > 0.55 else "#1e293b"
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Entrena el clasificador BasuraFix.")
    parser.add_argument("--data", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12, help="Épocas de fine-tuning.")
    parser.add_argument("--lr", type=float, default=3e-4, help="LR del backbone en fine-tuning.")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="LR de la cabeza (fase 1).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "basurafix_mobilenetv3.pt")
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = _device(args.device)
    amp = device.type == "cuda"

    if not args.data.exists():
        print(
            f"No existe {args.data}. Ejecuta primero:\n"
            "  python -m basurafix download\n  python -m basurafix prepare",
            file=sys.stderr,
        )
        return 1

    print("=" * 66)
    print(" BasuraFix · entrenamiento CNN")
    print(f" Arquitectura : {args.arch}")
    print(f" Dispositivo  : {device}  ({args.threads} hilos)")
    print(f" Imagen       : {args.img_size}x{args.img_size}  |  batch {args.batch_size}")
    print(f" Épocas       : {args.epochs_head} cabeza + {args.epochs} fine-tuning")
    print("=" * 66)

    loaders, datasets, counts = build_dataloaders(
        args.data, args.img_size, args.batch_size, args.workers, CLASSES
    )
    if "train" not in loaders or "val" not in loaders:
        print("Faltan splits de train/val en el manifest.", file=sys.stderr)
        return 1
    if "test" not in loaders:
        print("[aviso] No hay split de test; se usará validación para el reporte final.")

    train_labels = datasets["train"].labels
    weights = _class_weights(train_labels)
    print("\nImágenes por split:")
    for split, dataset in datasets.items():
        print(f"  {split:5}: {len(dataset):>6,}")
    print("Pesos de clase:", {c: round(float(w), 2) for c, w in zip(CLASSES, weights)})

    model = build_model(
        args.arch, num_classes=len(CLASSES), pretrained=not args.no_pretrained, dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=args.label_smoothing
    )

    history: list[dict] = []
    best_f1 = -1.0
    best_state: dict | None = None
    best_epoch = 0
    best_val: dict = {}
    started = time.perf_counter()
    epoch_counter = 0

    # ------------------------- Fase 1: cabeza -------------------------
    freeze_backbone(model)
    print(f"\nFase 1 · cabeza con backbone congelado ({count_trainable(model):,} parámetros)")
    optimizer = build_optimizer(model, lr=args.lr_head, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs_head, 1), eta_min=args.lr_head * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None

    for epoch in range(1, args.epochs_head + 1):
        epoch_counter += 1
        t0 = time.perf_counter()
        train_metrics = _run_epoch(
            model, loaders["train"], criterion, device, optimizer, phase1=True, amp=amp
        )
        val_metrics = _run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]
        row = {
            "epoch": epoch_counter,
            "phase": "head",
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "lr": lr,
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        print(
            f"[{epoch_counter:02d}/{args.epochs_head + args.epochs}] cabeza  "
            f"loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
            f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} F1 {row['val_f1']:.3f} "
            f"({row['seconds']:.0f}s)"
        )
        if row["val_f1"] > best_f1:
            best_f1 = row["val_f1"]
            best_epoch = epoch_counter
            best_val = dict(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---------------------- Fase 2: fine-tuning -----------------------
    unfreeze_all(model)
    print(f"\nFase 2 · fine-tuning completo ({count_trainable(model):,} parámetros)")
    optimizer = build_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, head_lr_multiplier=2.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_counter += 1
        t0 = time.perf_counter()
        train_metrics = _run_epoch(
            model, loaders["train"], criterion, device, optimizer, amp=amp
        )
        val_metrics = _run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch_counter,
            "phase": "finetune",
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "lr": lr,
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        improved = row["val_f1"] > best_f1 + 1e-4
        marker = "  ← mejor" if improved else ""
        print(
            f"[{epoch_counter:02d}/{args.epochs_head + args.epochs}] fine-tune "
            f"loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
            f"val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} F1 {row['val_f1']:.3f} "
            f"({row['seconds']:.0f}s){marker}"
        )
        if improved:
            best_f1 = row["val_f1"]
            best_epoch = epoch_counter
            best_val = dict(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping: sin mejora en {args.patience} épocas.")
                break

    if best_state is None:
        print("No se obtuvo ningún checkpoint válido.", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    model.eval()
    train_seconds = time.perf_counter() - started

    # ------------------------- Evaluación ------------------------------
    report_split = "test" if "test" in loaders else "val"
    test_loader = loaders[report_split]
    print(f"\nEvaluando el mejor checkpoint (época {best_epoch}) en {report_split}...")
    test_metrics = _run_epoch(model, test_loader, criterion, device)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, targets, _ in test_loader:
            logits = model(images.to(device))
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    tta_acc, tta_f1 = _evaluate_tta(model, test_loader, device)
    report_txt = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=[CLASS_NAMES_ES[c] for c in CLASSES],
        digits=4,
        zero_division=0,
    )
    print("\n" + report_txt)
    print(f"Precisión test: {test_metrics['accuracy']:.2%}  |  F1 macro: {test_metrics['macro_f1']:.4f}")
    print(f"Top-3 test: {test_metrics['top3_accuracy']:.2%}")
    print(f"Con TTA (espejo): {tta_acc:.2%}  |  F1 macro: {tta_f1:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    cm = _save_confusion(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")
    _save_history_plot(history, args.epochs_head, REPORTS_DIR / "training_history.png")

    stats = read_json(STATS_PATH, default={}) or {}
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arch": args.arch,
        "img_size": args.img_size,
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        "val_accuracy": best_val.get("val_acc"),
        "val_macro_f1": best_val.get("val_f1"),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_top3_accuracy": test_metrics["top3_accuracy"],
        "test_accuracy_tta": tta_acc,
        "test_macro_f1_tta": tta_f1,
        "test_split": report_split,
        "confusion_matrix": cm.tolist(),
        "class_counts": {
            cls: stats.get("classes", {}).get(cls, {}) for cls in CLASSES
        },
        "total_images": stats.get("total_images"),
        "history": history,
    }
    write_json(REPORTS_DIR / "metrics.json", metrics)
    write_json(REPORTS_DIR / "classification_report.json", {
        "text": report_txt,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    payload = checkpoint_payload(
        model,
        arch=args.arch,
        img_size=args.img_size,
        metrics={
            "val_accuracy": best_val.get("val_acc"),
            "val_macro_f1": best_val.get("val_f1"),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_accuracy_tta": tta_acc,
            "test_macro_f1_tta": tta_f1,
            "test_split": report_split,
            "best_epoch": best_epoch,
            "train_seconds": round(train_seconds, 1),
        },
        history=history,
        sources=stats.get("sources", {}),
        dataset={
            "total_images": stats.get("total_images"),
            "splits": {
                split: len(dataset) for split, dataset in datasets.items()
            },
            "classes": stats.get("classes", {}),
        },
        extra={
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "version": "1.0.0",
            "tta": True,
        },
    )
    torch.save(payload, args.out)
    print(f"\nModelo guardado en {args.out}")
    print(f"Reportes en {REPORTS_DIR}")
    print(f"Tiempo total: {format_duration(train_seconds)}")
    print("\nSiguiente paso:  python -m basurafix gui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
