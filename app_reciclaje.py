"""BasuraFix · Programa todo-en-uno de clasificación de residuos con IA y Webcam.

Usa los pesos entrenados (`basurafix_mobilenetv3.pt`, 96.34% de precisión en test con TTA)
sobre 8 clases unificadas (TrashNet + RealWaste + Garbage12).

Incluye:
  - Interfaz gráfica (Tkinter) para subir/arrastrar imágenes
  - Soporte de WEBCAM en vivo (clasificación en tiempo real + captura de foto)
  - TTA (Test-Time Augmentation con espejo horizontal)
  - Mapas de calor Grad-CAM para explicar qué mira la red neuronal
  - Uso por línea de comandos para clasificar imágenes directamente

Uso:
  py app_reciclaje.py                      # Abre la interfaz gráfica (Fotos + Webcam)
  py app_reciclaje.py --webcam             # Abre la interfaz e inicia la webcam directamente
  py app_reciclaje.py mi_foto.jpg          # Clasifica una imagen por consola
  py app_reciclaje.py mi_foto.jpg --explain gradcam.png
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps, ImageTk
from torchvision import models, transforms

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:
    pass

# ===========================================================================
# 1. CONFIGURACIÓN Y TAXONOMÍA (8 CLASES)
# ===========================================================================
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    APP_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_DIR

BASE_DIR = APP_DIR


def _resolve_resource(filename: str) -> Path:
    """Busca primero junto al .exe/.py y luego dentro del bundle PyInstaller."""
    external = APP_DIR / filename
    if external.exists():
        return external
    bundled = BUNDLE_DIR / filename
    if bundled.exists():
        return bundled
    return external


DEFAULT_MODEL_PATH = _resolve_resource("basurafix_mobilenetv3.pt")

CLASSES: list[str] = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "textile",
    "trash",
]

CLASS_INFO: dict[str, dict[str, str]] = {
    "cardboard": {
        "es": "Cartón",
        "color": "#3b82f6",
        "container": "Contenedor azul · Papel y cartón",
        "tip": "Aplana las cajas, retira cintas y restos de comida. Mantenlo seco.",
    },
    "glass": {
        "es": "Vidrio",
        "color": "#22c55e",
        "container": "Contenedor verde · Vidrio",
        "tip": "Enjuaga botellas y frascos. No mezcles cerámica, espejos ni focos.",
    },
    "metal": {
        "es": "Metal",
        "color": "#f59e0b",
        "container": "Contenedor amarillo · Plásticos y metales",
        "tip": "Enjuaga y aplana latas. Las tapas de metal también se reciclan.",
    },
    "paper": {
        "es": "Papel",
        "color": "#60a5fa",
        "container": "Contenedor azul · Papel y cartón",
        "tip": "Mantén el papel seco y limpio. No incluyas papel higiénico usado.",
    },
    "plastic": {
        "es": "Plástico",
        "color": "#eab308",
        "container": "Contenedor amarillo · Plásticos y metales",
        "tip": "Enjuaga y compacta botellas y envases. Revisa el código de reciclaje.",
    },
    "organic": {
        "es": "Orgánico",
        "color": "#b45309",
        "container": "Contenedor marrón · Residuos orgánicos",
        "tip": "Restos de comida y vegetación. Ideal para compostaje doméstico.",
    },
    "textile": {
        "es": "Textil",
        "color": "#a78bfa",
        "container": "Punto de acopio textil",
        "tip": "Ropa, telas y calzado en buen estado: dona o lleva a un punto de acopio.",
    },
    "trash": {
        "es": "Basura general",
        "color": "#94a3b8",
        "container": "Contenedor negro/gris · No reciclable",
        "tip": "Residuo no reciclable. Reduce y reutiliza antes de desecharlo.",
    },
}

CLASS_NAMES_ES = {k: v["es"] for k, v in CLASS_INFO.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UNCERTAIN_CONFIDENCE = 0.45
UNCERTAIN_MARGIN = 0.10


# ===========================================================================
# 2. MODELO, GRAD-CAM E INFERENCIA (CON TTA)
# ===========================================================================
def setup_console() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_image(path: str | Path) -> Image.Image:
    """Abre una imagen, corrige orientación EXIF y convierte transparencias a RGB."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, (17, 28, 46))
            rgba = img.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            return background
        return img.convert("RGB")


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (17, 28, 46))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def build_model(
    arch: str = "mobilenet_v3_large",
    num_classes: int = len(CLASSES),
    pretrained: bool = False,
    dropout: float = 0.25,
) -> nn.Module:
    arch = arch.lower()
    weights = "DEFAULT" if pretrained else None
    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout)
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout)
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, num_classes))
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")
    return model


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de pesos {path}. Ejecuta el notebook TrashNet_Clasificacion.ipynb para generarlo."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", checkpoint.get("clases", CLASSES))
    arch = checkpoint.get("arch", "mobilenet_v3_large")
    model = build_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


class GradCAM:
    def __init__(self, model: torch.nn.Module, arch: str):
        self.model = model
        self.arch = arch
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = []
        layer = self._target_layer(model, arch)
        self._handles.append(layer.register_forward_hook(self._forward_hook))
        self._handles.append(layer.register_full_backward_hook(self._backward_hook))

    @staticmethod
    def _target_layer(model: torch.nn.Module, arch: str):
        if arch.startswith(("mobilenet", "efficientnet", "convnext")):
            return model.features[-1]
        if arch.startswith("resnet"):
            return model.layer4[-1]
        raise ValueError(f"Grad-CAM no soportado para {arch}")

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(self, tensor: torch.Tensor, class_index: int | None = None):
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        logits = self.model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_index


def _apply_turbo_colormap(cam_uint8: np.ndarray) -> np.ndarray:
    """Convierte un mapa 2D [0..255] en RGB usando Turbo (OpenCV o fallback NumPy)."""
    try:
        import cv2

        bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        t = cam_uint8.astype(np.float32) / 255.0
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45, size: int = 512
) -> tuple[Image.Image, Image.Image]:
    base = image.convert("RGB")
    width, height = base.size
    scale = size / max(width, height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    base = base.resize(new_size, Image.Resampling.LANCZOS)

    cam_img = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        new_size, Image.Resampling.BILINEAR
    )
    cam_uint8 = np.asarray(cam_img, dtype=np.uint8)
    heat = _apply_turbo_colormap(cam_uint8).astype(np.float32)
    original = np.asarray(base, dtype=np.float32)
    blended = (original * (1.0 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return base, Image.fromarray(blended)


class BasuraPredictor:
    """Carga el checkpoint entrenado y realiza predicciones con TTA y Grad-CAM."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.checkpoint = load_checkpoint(model_path, self.device)
        self.classes: list[str] = self.checkpoint.get("classes", CLASSES)
        self.class_names_es: dict[str, str] = self.checkpoint.get("class_names_es", CLASS_NAMES_ES)
        self.class_info: dict[str, dict] = self.checkpoint.get("class_info", CLASS_INFO)
        self.img_size = int(self.checkpoint.get("img_size", 224))
        mean = self.checkpoint.get("mean", IMAGENET_MEAN)
        std = self.checkpoint.get("std", IMAGENET_STD)
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(round(self.img_size * 1.14)), antialias=True),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self._gradcam: GradCAM | None = None
        self._lock = threading.Lock()

    def predict(self, image: Image.Image, topk: int = 3, tta: bool = True) -> dict:
        started = time.perf_counter()
        image = _ensure_rgb(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with self._lock, torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)
            if tta:
                flipped = torch.flip(tensor, dims=[3])
                probabilities = (probabilities + torch.softmax(self.model(flipped), dim=1)) / 2.0
        probabilities = probabilities[0].cpu()

        values, indices = torch.topk(probabilities, k=min(topk, len(self.classes)))
        top = []
        for value, index in zip(values.tolist(), indices.tolist()):
            label = self.classes[index]
            info = self.class_info.get(label, {})
            top.append(
                {
                    "label": label,
                    "label_es": self.class_names_es.get(label, label),
                    "prob": float(value),
                    "percent": float(value) * 100.0,
                    "color": info.get("color", "#94a3b8"),
                    "container": info.get("container", ""),
                }
            )

        confidence = top[0]["prob"]
        margin = top[0]["prob"] - (top[1]["prob"] if len(top) > 1 else 0.0)
        best = top[0]
        warning = None
        if confidence < UNCERTAIN_CONFIDENCE:
            warning = "Confianza baja. Prueba acercando el objeto con buena luz y fondo limpio."
        elif margin < UNCERTAIN_MARGIN:
            warning = "Resultado ambiguo: hay dos clases muy parecidas. Prueba desde otro ángulo."

        return {
            "label": best["label"],
            "label_es": best["label_es"],
            "confidence": confidence,
            "confidence_percent": best["percent"],
            "margin": margin,
            "top": top,
            "probabilities": {
                label: float(prob) for label, prob in zip(self.classes, probabilities.tolist())
            },
            "container": best["container"],
            "advice": self.class_info.get(best["label"], {}).get("tip", ""),
            "warning": warning,
            "uncertain": warning is not None,
            "tta": tta,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    def explain(self, image: Image.Image, size: int = 470) -> dict:
        image = _ensure_rgb(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._lock:
            if self._gradcam is None:
                self._gradcam = GradCAM(self.model, self.checkpoint.get("arch", "mobilenet_v3_large"))
            cam, class_index = self._gradcam.generate(tensor)
        original, heatmap = overlay_heatmap(image, cam, size=size)
        return {
            "original": original,
            "heatmap": heatmap,
            "class_index": class_index,
            "label": self.classes[class_index],
            "label_es": self.class_names_es.get(self.classes[class_index], ""),
        }

    def info(self) -> dict:
        metrics = self.checkpoint.get("metrics", {})
        dataset = self.checkpoint.get("dataset", {})
        return {
            "arch": self.checkpoint.get("arch", "mobilenet_v3_large"),
            "img_size": self.img_size,
            "classes": [self.class_names_es.get(c, c) for c in self.classes],
            "test_accuracy": metrics.get("test_accuracy"),
            "test_accuracy_tta": metrics.get("test_accuracy_tta"),
            "total_images": dataset.get("total_images"),
        }


# ===========================================================================
# 3. INTERFAZ GRÁFICA CON SOPORTE DE WEBCAM EN TIEMPO REAL
# ===========================================================================
BG = "#08111f"
CARD = "#101d30"
CARD_2 = "#16253c"
CARD_3 = "#0c1728"
BORDER = "#223552"
TEXT = "#eef4ff"
MUTED = "#93a7c4"
ACCENT = "#34d399"
ACCENT_DARK = "#052e22"
BLUE = "#38bdf8"
AMBER = "#fbbf24"
RED = "#f87171"

ARCH_NAMES = {
    "mobilenet_v3_large": "MobileNetV3-Large",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b1": "EfficientNet-B1",
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet50": "ResNet-50",
    "resnet18": "ResNet-18",
}

IMAGE_TYPES = [
    ("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp *.avif *.heic *.heif"),
    ("Todos los archivos", "*.*"),
]


def _contrast_text(hex_color: str) -> str:
    try:
        value = hex_color.lstrip("#")
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "#08111f" if luminance > 0.62 else "#ffffff"
    except Exception:
        return "#ffffff"


def _shorten(path: str, max_chars: int = 48) -> str:
    if len(path) <= max_chars:
        return path
    return "…" + path[-(max_chars - 1) :]


def _make_button(parent, text, command, *, bg, fg, hover, font, padx=14, pady=9):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=hover,
        activeforeground=fg,
        disabledforeground="#5b6d87",
        relief="flat",
        bd=0,
        cursor="hand2",
        font=font,
        padx=padx,
        pady=pady,
        highlightthickness=0,
    )
    button._normal_bg = bg
    button._hover_bg = hover
    button.bind(
        "<Enter>",
        lambda e: button.configure(bg=button._hover_bg) if button["state"] != "disabled" else None,
    )
    button.bind(
        "<Leave>",
        lambda e: button.configure(bg=button._normal_bg) if button["state"] != "disabled" else None,
    )
    return button


class BasuraFixApp:
    def __init__(self, root: tk.Tk, *, demo_path: str | None = None, start_webcam: bool = False):
        self.root = root
        self.demo_path = demo_path
        self.start_webcam_on_load = start_webcam
        self.predictor: BasuraPredictor | None = None
        self._image: Image.Image | None = None
        self._image_path: str | None = None
        self._result: dict | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._heat_photos: list[ImageTk.PhotoImage] = []
        self._last_dir = str(Path.home() / "Downloads")
        self._busy = False
        self._model_ready = False
        self._top_probs = [0.0, 0.0, 0.0]
        self._top_colors = ["#334155"] * 3

        # Estado de Webcam
        self._webcam_active = False
        self._cap = None
        self._webcam_job = None
        self._webcam_predicting = False
        self._last_webcam_pred_time = 0.0
        self._release_thread: threading.Thread | None = None

        self.root.title("BasuraFix · Clasificador de Basura con IA + Webcam")
        self.root.geometry("1160x780")
        self.root.minsize(1020, 700)
        self.root.configure(bg=BG)
        icon_path = _resolve_resource("basurafix.ico")
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        self._init_fonts()
        self._build_ui()
        self._bind_keys()
        self._start_model_load()

    def _init_fonts(self) -> None:
        family = "Segoe UI"
        self.f_title = tkfont.Font(family=family, size=23, weight="bold")
        self.f_sub = tkfont.Font(family=family, size=10)
        self.f_section = tkfont.Font(family=family, size=8, weight="bold")
        self.f_body = tkfont.Font(family=family, size=10)
        self.f_body_bold = tkfont.Font(family=family, size=10, weight="bold")
        self.f_result = tkfont.Font(family=family, size=27, weight="bold")
        self.f_conf = tkfont.Font(family=family, size=29, weight="bold")
        self.f_button = tkfont.Font(family=family, size=10, weight="bold")
        self.f_small = tkfont.Font(family=family, size=8)
        self.f_badge = tkfont.Font(family=family, size=9, weight="bold")
        self.f_caption = tkfont.Font(family=family, size=11, weight="bold")
        self.f_rank = tkfont.Font(family=family, size=9, weight="bold")

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(4, 6))
        body.grid_columnconfigure(0, weight=0, minsize=520)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        self._build_left(left)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_right(right)

        self._build_footer()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 10))
        header.grid_columnconfigure(1, weight=1)

        logo = tk.Canvas(header, width=46, height=46, bg=BG, highlightthickness=0)
        logo.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))
        logo.create_oval(2, 2, 44, 44, fill=ACCENT, outline="")
        logo.create_text(23, 22, text="♻", fill=ACCENT_DARK, font=("Segoe UI Symbol", 21, "bold"))

        tk.Label(header, text="BasuraFix", bg=BG, fg=TEXT, font=self.f_title).grid(
            row=0, column=1, sticky="sw"
        )
        tk.Label(
            header,
            text="Clasificador inteligente de residuos · Imagen y Webcam en vivo (96.3% precisión TTA)",
            bg=BG,
            fg=MUTED,
            font=self.f_sub,
        ).grid(row=1, column=1, sticky="nw")

        badges = tk.Frame(header, bg=BG)
        badges.grid(row=0, column=2, rowspan=2, sticky="e")
        self.model_badge = tk.Label(
            badges,
            text="CARGANDO MODELO…",
            bg=CARD_2,
            fg=MUTED,
            font=self.f_badge,
            padx=14,
            pady=7,
        )
        self.model_badge.pack(side="right", padx=(8, 0))
        tk.Label(
            badges,
            text="8 CLASES · WEBCAM",
            bg="#123527",
            fg=ACCENT,
            font=self.f_badge,
            padx=12,
            pady=7,
        ).pack(side="right")

    def _build_left(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=CARD)
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        head.grid_columnconfigure(0, weight=1)
        tk.Label(head, text="VISTA PREVIA / CÁMARA", bg=CARD, fg=MUTED, font=self.f_section).grid(
            row=0, column=0, sticky="w"
        )
        self.path_var = tk.StringVar(value="Ninguna imagen seleccionada")
        tk.Label(
            head,
            textvariable=self.path_var,
            bg=CARD,
            fg="#6f86a5",
            font=self.f_small,
            anchor="w",
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.preview = tk.Canvas(
            parent,
            bg=CARD_3,
            highlightthickness=0,
            bd=0,
            width=480,
            height=390,
        )
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        self.preview.bind("<Configure>", lambda event: self._render_preview())

        controls = tk.Frame(parent, bg=CARD)
        controls.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        self.select_btn = _make_button(
            controls,
            "📁 Subir imagen",
            self.select_file,
            bg=ACCENT,
            fg=ACCENT_DARK,
            hover="#6ee7b7",
            font=self.f_button,
        )
        self.select_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.webcam_btn = _make_button(
            controls,
            "📷 Usar Webcam",
            self.toggle_webcam,
            bg=BLUE,
            fg="#08111f",
            hover="#7dd3fc",
            font=self.f_button,
        )
        self.webcam_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.heat_btn = _make_button(
            controls,
            "🔥 Grad-CAM",
            self.show_heatmap,
            bg=CARD_2,
            fg=TEXT,
            hover="#233854",
            font=self.f_button,
        )
        self.heat_btn.grid(row=0, column=2)
        self.heat_btn.configure(state="disabled")

        hint = "Atajos: Ctrl+O abrir foto · W activar/congelar Webcam · Espacio capturar"
        tk.Label(
            parent,
            text=hint,
            bg=CARD,
            fg="#6f86a5",
            font=self.f_small,
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))

        try:
            from tkinterdnd2 import DND_FILES

            self.preview.drop_target_register(DND_FILES)
            self.preview.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _build_right(self, parent: tk.Frame) -> None:
        result = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        result.grid(row=0, column=0, sticky="ew")
        result.grid_columnconfigure(0, weight=1)

        top = tk.Frame(result, bg=CARD)
        top.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 0))
        top.grid_columnconfigure(0, weight=1)
        tk.Label(top, text="RESULTADO", bg=CARD, fg=MUTED, font=self.f_section).grid(
            row=0, column=0, sticky="w"
        )
        self.time_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.time_var, bg=CARD, fg="#6f86a5", font=self.f_small).grid(
            row=0, column=1, sticky="e"
        )

        name_row = tk.Frame(result, bg=CARD)
        name_row.grid(row=1, column=0, sticky="ew", padx=22, pady=(6, 0))
        name_row.grid_columnconfigure(1, weight=1)
        self.dot = tk.Canvas(name_row, width=16, height=16, bg=CARD, highlightthickness=0)
        self.dot.grid(row=0, column=0, padx=(0, 10))
        self._dot_id = self.dot.create_oval(2, 2, 14, 14, fill="#334155", outline="")
        self.name_var = tk.StringVar(value="Sin analizar")
        self.name_label = tk.Label(
            name_row,
            textvariable=self.name_var,
            bg=CARD,
            fg=TEXT,
            font=self.f_result,
            anchor="w",
        )
        self.name_label.grid(row=0, column=1, sticky="w")
        self.conf_var = tk.StringVar(value="—")
        self.conf_label = tk.Label(
            name_row,
            textvariable=self.conf_var,
            bg=CARD,
            fg=MUTED,
            font=self.f_conf,
        )
        self.conf_label.grid(row=0, column=2, sticky="e")

        self.conf_bar = tk.Canvas(result, height=9, width=120, bg=CARD, highlightthickness=0)
        self.conf_bar.grid(row=2, column=0, sticky="ew", padx=22, pady=(10, 12))
        self.conf_bar.bind("<Configure>", lambda e: self._draw_confidence())

        self.container_label = tk.Label(
            result,
            text="Selecciona una imagen o activa la webcam para comenzar",
            bg=CARD_2,
            fg=MUTED,
            font=self.f_body_bold,
            padx=12,
            pady=7,
            anchor="w",
        )
        self.container_label.grid(row=3, column=0, sticky="ew", padx=22)

        self.tip_var = tk.StringVar(
            value="La CNN analizará la foto o el video de tu cámara en tiempo real y te dirá en qué contenedor depositar el residuo."
        )
        tk.Label(
            result,
            textvariable=self.tip_var,
            bg=CARD,
            fg=MUTED,
            font=self.f_body,
            wraplength=530,
            justify="left",
            anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=22, pady=(10, 0))

        self.warning_var = tk.StringVar(value="")
        self.warning_label = tk.Label(
            result,
            textvariable=self.warning_var,
            bg=CARD,
            fg=AMBER,
            font=self.f_small,
            wraplength=530,
            justify="left",
            anchor="w",
        )
        self.warning_label.grid(row=5, column=0, sticky="ew", padx=22, pady=(6, 0))
        self.warning_label.grid_remove()
        tk.Frame(result, bg=CARD, height=16).grid(row=6, column=0)

        top3 = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        top3.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        top3.grid_columnconfigure(0, weight=1)
        tk.Label(
            top3,
            text="PRINCIPALES COINCIDENCIAS (TOP-3)",
            bg=CARD,
            fg=MUTED,
            font=self.f_section,
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(15, 4))

        self.top_rows = []
        for index in range(3):
            row = tk.Frame(top3, bg=CARD)
            row.grid(row=index + 1, column=0, sticky="ew", padx=22, pady=5)
            row.grid_columnconfigure(2, weight=1)
            rank = tk.Label(
                row,
                text=str(index + 1),
                bg=CARD,
                fg="#58708f",
                font=self.f_rank,
                width=2,
                anchor="w",
            )
            rank.grid(row=0, column=0, sticky="w")
            name = tk.Label(
                row,
                text="—",
                bg=CARD,
                fg=MUTED,
                font=self.f_body_bold,
                width=14,
                anchor="w",
            )
            name.grid(row=0, column=1, sticky="w")
            bar = tk.Canvas(row, height=8, width=120, bg=CARD, highlightthickness=0)
            bar.grid(row=0, column=2, sticky="ew", padx=(8, 10))
            pct = tk.Label(
                row,
                text="—",
                bg=CARD,
                fg=MUTED,
                font=self.f_body_bold,
                width=6,
                anchor="e",
            )
            pct.grid(row=0, column=3, sticky="e")
            bar.bind("<Configure>", lambda e, i=index: self._draw_top_bar(i))
            self.top_rows.append({"name": name, "bar": bar, "pct": pct})
        tk.Frame(top3, bg=CARD, height=10).grid(row=4, column=0)

    def _build_footer(self) -> None:
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", padx=26, pady=(0, 14))
        footer.grid_columnconfigure(1, weight=1)

        self.status_dot = tk.Canvas(footer, width=10, height=10, bg=BG, highlightthickness=0)
        self.status_dot.grid(row=0, column=0, padx=(0, 7))
        self._status_id = self.status_dot.create_oval(1, 1, 9, 9, fill=AMBER, outline="")
        self.status_var = tk.StringVar(value="Cargando modelo…")
        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED, font=self.f_small).grid(
            row=0, column=1, sticky="w"
        )

        self.model_info_var = tk.StringVar(value="")
        tk.Label(
            footer,
            textvariable=self.model_info_var,
            bg=BG,
            fg="#58708f",
            font=self.f_small,
        ).grid(row=0, column=2, sticky="e")

    # ------------------------------------------------------------------
    # Carga de modelo
    # ------------------------------------------------------------------
    def _set_status(self, text: str, kind: str = "info") -> None:
        colors = {"info": MUTED, "ready": ACCENT, "busy": AMBER, "error": RED}
        self.status_var.set(text)
        self.status_dot.itemconfigure(self._status_id, fill=colors.get(kind, MUTED))

    def _start_model_load(self) -> None:
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self) -> None:
        started = time.perf_counter()
        try:
            predictor = BasuraPredictor(DEFAULT_MODEL_PATH)
            info = predictor.info()
            elapsed = time.perf_counter() - started
            self.root.after(0, lambda: self._on_model_loaded(predictor, info, elapsed))
        except Exception as exc:
            self.root.after(0, lambda: self._on_model_error(exc))

    def _on_model_loaded(self, predictor: BasuraPredictor, info: dict, elapsed: float) -> None:
        self.predictor = predictor
        self._model_ready = True
        arch = ARCH_NAMES.get(info.get("arch", ""), info.get("arch", "CNN"))
        self.model_badge.configure(text=arch.upper(), fg=ACCENT, bg="#123527")
        total = info.get("total_images")
        accuracy = info.get("test_accuracy_tta") or info.get("test_accuracy")
        parts = []
        if accuracy is not None:
            parts.append(f"Precisión test {accuracy * 100:.2f}%")
        if total:
            parts.append(f"{total:,} imágenes")
        parts.append(f"cargado en {elapsed:.1f} s")
        self.model_info_var.set(" · ".join(parts))
        self._set_status("Modelo listo. Sube una foto o pulsa «Usar Webcam».", "ready")
        if self.demo_path:
            self.root.after(200, lambda: self.load_path(self.demo_path))
        elif self.start_webcam_on_load:
            self.root.after(200, self.toggle_webcam)

    def _on_model_error(self, exc: Exception) -> None:
        self._model_ready = False
        self._set_status("No se pudo cargar el modelo", "error")
        self.model_badge.configure(text="MODELO NO DISPONIBLE", fg=RED, bg="#3b1218")
        self.tip_var.set(f"No se encontró {DEFAULT_MODEL_PATH.name}. Ejecuta el notebook primero.")
        self.model_info_var.set(str(exc)[:90])

    # ------------------------------------------------------------------
    # Webcam en tiempo real
    # ------------------------------------------------------------------
    def toggle_webcam(self) -> None:
        if not self._model_ready:
            messagebox.showwarning("Espera", "El modelo aún está cargando.", parent=self.root)
            return
        if self._webcam_active:
            self.stop_webcam(keep_frame=True)
        else:
            self.start_webcam()

    def start_webcam(self) -> None:
        try:
            import cv2
        except ImportError:
            messagebox.showerror(
                "OpenCV requerido",
                "Para usar la webcam instala opencv-python:\n  py -m pip install opencv-python",
                parent=self.root,
            )
            return

        self.stop_webcam(keep_frame=False, analyze=False)
        if self._release_thread is not None and self._release_thread.is_alive():
            self._release_thread.join(timeout=1.5)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            messagebox.showerror(
                "Webcam no disponible",
                "No se pudo abrir ninguna cámara web conectada al equipo.",
                parent=self.root,
            )
            return

        self._cap = cap
        self._webcam_active = True
        self.path_var.set("🔴 EN VIVO · Webcam activa")
        self.webcam_btn._normal_bg = AMBER
        self.webcam_btn._hover_bg = "#fcd34d"
        self.webcam_btn.configure(text="📸 Capturar / Pausar", bg=AMBER, activebackground="#fcd34d")
        self._set_status("Webcam en vivo · Clasificando residuos automáticamente…", "ready")
        self._webcam_loop()

    def stop_webcam(self, keep_frame: bool = True, analyze: bool = True) -> None:
        was_active = self._webcam_active
        self._webcam_active = False
        if self._webcam_job is not None:
            try:
                self.root.after_cancel(self._webcam_job)
            except Exception:
                pass
            self._webcam_job = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            t = threading.Thread(target=cap.release, daemon=True)
            t.start()
            self._release_thread = t
        self.webcam_btn._normal_bg = BLUE
        self.webcam_btn._hover_bg = "#7dd3fc"
        self.webcam_btn.configure(text="📷 Usar Webcam", bg=BLUE, activebackground="#7dd3fc")
        if was_active and keep_frame and self._image is not None:
            self.path_var.set("Fotograma capturado de Webcam")
            if analyze:
                self._set_status("Fotograma de webcam capturado. Puedes ver su Grad-CAM.", "ready")
                self._analyze()

    def _webcam_loop(self) -> None:
        if not self._webcam_active or self._cap is None:
            return
        import cv2

        ok, frame = self._cap.read()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)  # Efecto espejo natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self._image = pil_img
            self._render_preview()

            now = time.perf_counter()
            if (
                not self._webcam_predicting
                and self.predictor is not None
                and (now - self._last_webcam_pred_time) >= 0.35
            ):
                self._webcam_predicting = True
                self._last_webcam_pred_time = now
                threading.Thread(
                    target=self._webcam_predict_worker, args=(pil_img.copy(),), daemon=True
                ).start()

        self._webcam_job = self.root.after(33, self._webcam_loop)

    def _webcam_predict_worker(self, image: Image.Image) -> None:
        try:
            if self.predictor is not None:
                result = self.predictor.predict(image, topk=3, tta=False)
                if self._webcam_active:
                    self.root.after(0, lambda: self._show_result(result, from_webcam=True))
        except Exception:
            pass
        finally:
            self._webcam_predicting = False

    # ------------------------------------------------------------------
    # Selección de archivo y análisis
    # ------------------------------------------------------------------
    def select_file(self) -> None:
        if not self._model_ready:
            return
        if self._webcam_active:
            self.stop_webcam(keep_frame=False)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona una foto de basura",
            initialdir=self._last_dir,
            filetypes=IMAGE_TYPES,
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        if self._webcam_active:
            self.stop_webcam(keep_frame=False)
        try:
            image = load_image(path)
        except Exception as exc:
            messagebox.showerror(
                "No se pudo abrir la imagen",
                f"El archivo no es una imagen válida:\n\n{exc}",
                parent=self.root,
            )
            return
        self._image = image
        self._image_path = str(path)
        self._last_dir = str(Path(path).parent)
        self.path_var.set(_shorten(Path(path).name, 48))
        self._render_preview()
        self._analyze()

    def _analyze(self) -> None:
        if self._image is None or self.predictor is None:
            return
        image = self._image
        self._set_busy(True, "Analizando imagen con la CNN…")
        threading.Thread(target=self._analysis_worker, args=(image,), daemon=True).start()

    def _analysis_worker(self, image: Image.Image) -> None:
        try:
            result = self.predictor.predict(image, topk=3, tta=True)
            self.root.after(0, lambda: self._show_result(result, from_webcam=False))
        except Exception as exc:
            self.root.after(0, lambda: self._show_analysis_error(exc))

    def _show_result(self, result: dict, from_webcam: bool = False) -> None:
        self._result = result
        if not from_webcam:
            self._set_busy(False)
        name = result["label_es"]
        color = result["top"][0]["color"]
        self.name_var.set(name)
        self.name_label.configure(fg=TEXT)
        self.conf_var.set(f"{result['confidence_percent']:.1f}%")
        self.conf_label.configure(fg=color)
        self.dot.itemconfigure(self._dot_id, fill=color)
        self.container_label.configure(
            text=result["container"] or "Residuo identificado",
            bg=color,
            fg=_contrast_text(color),
        )
        self.tip_var.set(result.get("advice") or "")
        mode_tag = "EN VIVO" if from_webcam else "TTA"
        self.time_var.set(f"{result['elapsed_ms']:.0f} ms · {mode_tag}")
        if result.get("warning"):
            self.warning_var.set("⚠  " + result["warning"])
            self.warning_label.grid()
        else:
            self.warning_label.grid_remove()
        self._top_probs = [item["prob"] for item in result["top"]]
        self._top_colors = [item["color"] for item in result["top"]]
        for index, item in enumerate(result["top"]):
            row = self.top_rows[index]
            row["name"].configure(text=item["label_es"], fg=TEXT if index == 0 else MUTED)
            row["pct"].configure(text=f"{item['percent']:.1f}%", fg=color if index == 0 else MUTED)
            self._draw_top_bar(index)
        self._draw_confidence()
        self.heat_btn.configure(state="normal")
        if not from_webcam:
            self._set_status(
                f"Listo · {result['label_es']} detectado con {result['confidence_percent']:.1f}% de confianza",
                "ready",
            )

    def _show_analysis_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self._set_status("Error al analizar la imagen", "error")
        messagebox.showerror("Error", f"No se pudo analizar la imagen:\n\n{exc}", parent=self.root)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        if busy:
            self.select_btn.configure(state="disabled")
            self.heat_btn.configure(state="disabled")
            if message:
                self._set_status(message, "busy")
            self.root.configure(cursor="watch")
        else:
            self.root.configure(cursor="")
            if self._model_ready:
                self.select_btn.configure(state="normal")
            if self._result:
                self.heat_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # Vista previa y gráficos
    # ------------------------------------------------------------------
    def _render_preview(self) -> None:
        canvas = self.preview
        canvas.delete("all")
        width = max(canvas.winfo_width(), 10)
        height = max(canvas.winfo_height(), 10)
        if self._image is None:
            canvas.create_text(
                width / 2,
                height / 2 - 26,
                text="Arrastra una foto o usa la Webcam",
                fill="#7c93b3",
                font=(self.f_caption.cget("family"), 14, "bold"),
            )
            canvas.create_text(
                width / 2,
                height / 2 + 6,
                text="Pulsa «Subir imagen» o «Usar Webcam»",
                fill="#5f7798",
                font=self.f_body,
            )
            canvas.create_text(
                width / 2,
                height / 2 + 42,
                text="JPG · PNG · WEBP · AVIF · HEIC · CÁMARA EN VIVO",
                fill="#3f5777",
                font=self.f_small,
            )
            return

        margin = 14
        box = (max(10, width - margin * 2), max(10, height - margin * 2))
        preview = self._image.copy()
        preview.thumbnail(box, Image.Resampling.BILINEAR if self._webcam_active else Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        canvas.create_image(width / 2, height / 2, image=self._preview_photo)
        border_col = ACCENT if self._webcam_active else "#2b4366"
        canvas.create_rectangle(
            width / 2 - preview.width / 2,
            height / 2 - preview.height / 2,
            width / 2 + preview.width / 2,
            height / 2 + preview.height / 2,
            outline=border_col,
            width=2 if self._webcam_active else 1,
        )

    def _draw_confidence(self) -> None:
        canvas = self.conf_bar
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_rectangle(0, 0, width, height, fill=CARD_2, outline="")
        if self._result:
            value = max(0.0, min(1.0, self._result["confidence"]))
            color = self._result["top"][0]["color"]
            canvas.create_rectangle(0, 0, width * value, height, fill=color, outline="")

    def _draw_top_bar(self, index: int) -> None:
        if index >= len(self.top_rows):
            return
        canvas = self.top_rows[index]["bar"]
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        canvas.create_rectangle(0, 0, width, height, fill=CARD_2, outline="")
        probability = self._top_probs[index] if index < len(self._top_probs) else 0.0
        color = self._top_colors[index] if index < len(self._top_colors) else "#334155"
        filled = width * max(0.0, min(1.0, probability))
        if filled > 0:
            canvas.create_rectangle(0, 0, filled, height, fill=color, outline="")

    # ------------------------------------------------------------------
    # Ventana de Mapa de Calor (Grad-CAM)
    # ------------------------------------------------------------------
    def show_heatmap(self) -> None:
        if self._image is None or self.predictor is None or self._busy:
            return
        if self._webcam_active:
            self.stop_webcam(keep_frame=True, analyze=False)
        self._set_busy(True, "Generando mapa de calor (Grad-CAM)…")
        threading.Thread(target=self._heatmap_worker, args=(self._image.copy(),), daemon=True).start()

    def _heatmap_worker(self, image: Image.Image) -> None:
        try:
            explanation = self.predictor.explain(image, size=470)
            self.root.after(0, lambda: self._show_heatmap_window(explanation))
        except Exception as exc:
            self.root.after(0, lambda: self._show_analysis_error(exc))

    def _show_heatmap_window(self, explanation: dict) -> None:
        self._set_busy(False)
        if self._result:
            self._set_status(
                f"Listo · Mapa de calor generado para «{explanation['label_es']}»",
                "ready",
            )
        else:
            self._set_status("Listo · Mapa de calor generado.", "ready")
        window = tk.Toplevel(self.root)
        window.title("BasuraFix · Mapa de calor Grad-CAM")
        window.configure(bg=BG)
        window.geometry("1010x610")
        window.minsize(900, 540)
        window.transient(self.root)
        window.bind("<Escape>", lambda event: window.destroy())

        header = tk.Frame(window, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(header, text="¿Dónde mira la CNN?", bg=BG, fg=TEXT, font=self.f_title).pack(
            anchor="w"
        )
        tk.Label(
            header,
            text=f"Grad-CAM sobre la clase «{explanation['label_es']}» · las zonas rojas/amarillas influyen más",
            bg=BG,
            fg=MUTED,
            font=self.f_body,
        ).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(window, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=(8, 0))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        labels = ["Foto analizada", "Mapa de calor (Grad-CAM)"]
        self._heat_photos = []
        for column, (caption, key) in enumerate(zip(labels, ("original", "heatmap"))):
            tk.Label(body, text=caption, bg=BG, fg=MUTED, font=self.f_body_bold).grid(
                row=0, column=column, sticky="w", padx=(0 if column == 0 else 12, 0)
            )
            frame = tk.Frame(body, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            frame.grid(
                row=1,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 12, 0),
                pady=(6, 0),
            )
            photo = ImageTk.PhotoImage(explanation[key])
            self._heat_photos.append(photo)
            tk.Label(frame, image=photo, bg=CARD).pack(expand=True, padx=6, pady=6)

        tk.Label(
            window,
            text="El mapa resalta las regiones que más aportaron a la decisión de la red neuronal.",
            bg=BG,
            fg="#6f86a5",
            font=self.f_small,
        ).pack(fill="x", padx=24, pady=(10, 14))

    def _on_drop(self, event) -> None:
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if paths:
            path = paths[0].strip("{}")
            if Path(path).is_file():
                self.load_path(path)

    def _bind_keys(self) -> None:
        self.root.bind("<Control-o>", lambda event: self.select_file())
        self.root.bind("<Control-O>", lambda event: self.select_file())
        self.root.bind("<w>", lambda event: self.toggle_webcam())
        self.root.bind("<W>", lambda event: self.toggle_webcam())
        self.root.bind("<space>", lambda event: self.toggle_webcam() if self._webcam_active else None)
        self.root.bind("<F5>", lambda event: self._analyze())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._webcam_active = False
        if self._webcam_job is not None:
            try:
                self.root.after_cancel(self._webcam_job)
            except Exception:
                pass
            self._webcam_job = None
        cap = self._cap
        self._cap = None
        if cap is not None:
            threading.Thread(target=cap.release, daemon=True).start()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        import os

        os._exit(0)


# ===========================================================================
# 4. PUNTO DE ENTRADA (INTERFAZ GRÁFICA + CLI)
# ===========================================================================
def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _print_cli_result(result: dict) -> None:
    print("\n" + "=" * 62)
    print(f"  {result['label_es'].upper()}  ·  {result['confidence_percent']:.1f}% de confianza")
    print("=" * 62)
    for index, item in enumerate(result["top"], start=1):
        bar = "#" * int(round(item["percent"] / 2.5))
        print(f"  {index}. {item['label_es']:<16} {item['percent']:6.1f}%  {bar}")
    print(f"\n  {result['container']}")
    if result["advice"]:
        print(f"  {result['advice']}")
    if result["warning"]:
        print(f"\n  [!] {result['warning']}")
    print(f"\n  Inferencia: {result['elapsed_ms']:.0f} ms · TTA: {'sí' if result['tta'] else 'no'}")


def _run_self_test(report_path: Path, model_path: Path) -> int:
    """Verifica modelo embebido, TTA, Grad-CAM, OpenCV y Tkinter/Drag&Drop."""
    report: dict = {"ok": False, "frozen": bool(getattr(sys, "frozen", False))}
    try:
        report["model_path"] = str(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        predictor = BasuraPredictor(model_path)
        report["model_info"] = predictor.info()

        # 1) Inferencia con TTA
        img = Image.fromarray(
            np.random.default_rng(42).integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
        )
        res = predictor.predict(img, topk=3, tta=True)
        report["predict_label"] = res["label_es"]
        report["predict_conf"] = round(res["confidence_percent"], 2)

        # 2) Grad-CAM + mapa de calor Turbo
        exp = predictor.explain(img, size=256)
        report["gradcam_size"] = list(exp["heatmap"].size)

        # 3) OpenCV (webcam backend disponible)
        import cv2

        report["cv2_version"] = cv2.__version__
        report["cv2_has_videocapture"] = hasattr(cv2, "VideoCapture")

        # 4) Tkinter + tkinterdnd2
        dnd_ok = False
        try:
            from tkinterdnd2 import TkinterDnD

            root = TkinterDnD.Tk()
            dnd_ok = True
        except Exception:
            root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        report["tkinter_ok"] = True
        report["tkinterdnd2_ok"] = dnd_ok

        report["ok"] = True
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(
        description="BasuraFix: Clasificador de residuos con IA (Interfaz Gráfica, Webcam y CLI)."
    )
    parser.add_argument("image", nargs="?", help="Ruta opcional de una imagen para clasificar por consola.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Ruta del archivo .pt")
    parser.add_argument("--webcam", action="store_true", help="Inicia la interfaz con la webcam encendida.")
    parser.add_argument("--demo", help="Abre la interfaz cargando una imagen de prueba.")
    parser.add_argument("--explain", type=Path, help="Guarda el mapa de calor Grad-CAM en esta ruta (modo CLI).")
    parser.add_argument("--no-tta", action="store_true", help="Desactiva Test-Time Augmentation en modo CLI.")
    parser.add_argument("--json", action="store_true", help="Devuelve salida JSON en modo CLI.")
    parser.add_argument("--self-test", type=Path, help="Ejecuta verificación completa y guarda reporte JSON.")
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test(args.self_test, args.model)

    # Si se pasa una imagen por argumento (sin --demo), se ejecuta por consola
    if args.image and not args.demo:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"No existe la imagen: {img_path}", file=sys.stderr)
            return 1
        predictor = BasuraPredictor(args.model)
        image = load_image(img_path)
        result = predictor.predict(image, topk=3, tta=not args.no_tta)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_cli_result(result)
        if args.explain:
            explanation = predictor.explain(image)
            args.explain.parent.mkdir(parents=True, exist_ok=True)
            comp = Image.new(
                "RGB",
                (explanation["original"].width * 2 + 12, explanation["original"].height),
                (15, 23, 42),
            )
            comp.paste(explanation["original"], (0, 0))
            comp.paste(explanation["heatmap"], (explanation["original"].width + 12, 0))
            comp.save(args.explain)
            print(f"\nGrad-CAM guardado en {args.explain}")
        return 0

    # Modo Interfaz Gráfica (Fotos + Webcam)
    _enable_dpi_awareness()
    try:
        try:
            from tkinterdnd2 import TkinterDnD

            root = TkinterDnD.Tk()
        except Exception:
            root = tk.Tk()
        BasuraFixApp(root, demo_path=args.demo, start_webcam=args.webcam)
        root.mainloop()
        return 0
    except Exception as exc:
        try:
            messagebox.showerror(
                "Error al iniciar BasuraFix",
                f"Ocurrió un error inesperado:\n\n{exc}\n\n{traceback.format_exc()[-600:]}",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
