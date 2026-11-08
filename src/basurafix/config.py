"""Configuración central de BasuraFix: rutas, clases y metadatos."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLES_DIR = BASE_DIR / "samples"

DEFAULT_MODEL_PATH = MODELS_DIR / "basurafix_mobilenetv3.pt"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
STATS_PATH = PROCESSED_DIR / "stats.json"

GARBAGE12_DIR = RAW_DIR / "garbage12"
GARBAGE12_ZIP = GARBAGE12_DIR / "garbage_classification.zip"
GARBAGE12_EXTRACT = GARBAGE12_DIR / "extracted"

# ---------------------------------------------------------------------------
# Taxonomía unificada (8 clases)
# ---------------------------------------------------------------------------
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

CLASS_NAMES_ES = {key: value["es"] for key, value in CLASS_INFO.items()}

# Frases que acompañan cada predicción en la interfaz.
CLASS_ADVICE = {
    key: f"{info['container']}. {info['tip']}" for key, info in CLASS_INFO.items()
}

# ---------------------------------------------------------------------------
# Mapeo de etiquetas de las fuentes originales a la taxonomía unificada
# ---------------------------------------------------------------------------
TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

REALWASTE_MAP = {
    "cardboard": "cardboard",
    "food organics": "organic",
    "glass": "glass",
    "metal": "metal",
    "miscellaneous trash": "trash",
    "paper": "paper",
    "plastic": "plastic",
    "textile trash": "textile",
    "vegetation": "organic",
}

# Alias aceptados para datasets adicionales (carpeta por clase).
LABEL_ALIASES = {
    # cartón
    "cardboard": "cardboard",
    "carton": "cardboard",
    "cartón": "cardboard",
    "cartones": "cardboard",
    # vidrio
    "glass": "glass",
    "vidrio": "glass",
    "vidrios": "glass",
    # metal
    "metal": "metal",
    "metals": "metal",
    "metales": "metal",
    "lata": "metal",
    "latas": "metal",
    "aluminum": "metal",
    "aluminium": "metal",
    # papel
    "paper": "paper",
    "papel": "paper",
    "papers": "paper",
    # plástico
    "plastic": "plastic",
    "plastics": "plastic",
    "plastico": "plastic",
    "plástico": "plastic",
    "plasticos": "plastic",
    "plásticos": "plastic",
    "pet": "plastic",
    # orgánico
    "organic": "organic",
    "organics": "organic",
    "organico": "organic",
    "orgánico": "organic",
    "food organics": "organic",
    "food_organics": "organic",
    "food waste": "organic",
    "biological": "organic",
    "biodegradable": "organic",
    "vegetation": "organic",
    "vegetal": "organic",
    "compost": "organic",
    # textil
    "textile": "textile",
    "textiles": "textile",
    "textil": "textile",
    "textile trash": "textile",
    "textile_trash": "textile",
    "clothes": "textile",
    "clothing": "textile",
    "shoes": "textile",
    "ropa": "textile",
    # basura general
    "trash": "trash",
    "miscellaneous trash": "trash",
    "miscellaneous_trash": "trash",
    "misc trash": "trash",
    "misc": "trash",
    "general": "trash",
    "general trash": "trash",
    "basura": "trash",
    "non recyclable": "trash",
    "non_recyclable": "trash",
    "residual": "trash",
    # clases que no queremos entrenar (peligrosas, etc.)
    "battery": None,
    "batteries": None,
    "hazardous": None,
    "e-waste": None,
    "electronic": None,
    "medical": None,
}


def normalize_label(label: str) -> str:
    """Normaliza una etiqueta: minúsculas, sin acentos, espacios -> '_'."""
    label = unicodedata.normalize("NFKD", str(label))
    label = "".join(ch for ch in label if not unicodedata.combining(ch))
    label = label.strip().lower()
    label = re.sub(r"[\s\-]+", "_", label)
    return label


def map_label(label: str) -> str | None:
    """Devuelve la clase unificada o None si la etiqueta no se reconoce."""
    raw = str(label).strip().lower()
    compact = re.sub(r"[\s\-]+", " ", raw)
    normalized = normalize_label(label)

    # 1) coincidencia exacta con alias (con espacios y con guiones bajos)
    for candidate in (compact, normalized, normalized.replace("_", " ")):
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]

    # 2) el nombre de la carpeta puede traer prefijos/sufijos
    for alias, target in LABEL_ALIASES.items():
        if len(alias) >= 4 and (alias in compact or alias.replace(" ", "_") in normalized):
            return target
    return None


# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_ARCH = "mobilenet_v3_large"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42
