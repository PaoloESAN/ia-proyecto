"""Interfaz gráfica de BasuraFix (Tkinter).

Uso:
    python -m basurafix gui
    python -m basurafix gui --demo samples/botella_plastico.jpg
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

from PIL import Image, ImageTk

from .config import DEFAULT_MODEL_PATH, SAMPLES_DIR
from .dataset import load_image
from .predict import BasuraPredictor
from .utils import setup_console

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
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


def _shorten(path: str, max_chars: int = 58) -> str:
    if len(path) <= max_chars:
        return path
    return "…" + path[-(max_chars - 1) :]


def _make_button(parent, text, command, *, bg, fg, hover, font, padx=16, pady=9):
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
    button.bind("<Enter>", lambda e: button.configure(bg=hover) if button["state"] != "disabled" else None)
    button.bind("<Leave>", lambda e: button.configure(bg=bg) if button["state"] != "disabled" else None)
    return button


class BasuraFixApp:
    def __init__(self, root: tk.Tk, *, demo_path: str | None = None, screenshot: str | None = None):
        self.root = root
        self.demo_path = demo_path
        self.screenshot_path = screenshot
        self.predictor: BasuraPredictor | None = None
        self._image: Image.Image | None = None
        self._image_path: str | None = None
        self._result: dict | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._heat_photos: list[ImageTk.PhotoImage] = []
        self._last_dir = str(Path.home() / "Downloads")
        self._busy = False
        self._model_ready = False
        self._status_kind = "info"
        self._top_probs = [0.0, 0.0, 0.0]
        self._top_colors = ["#334155"] * 3

        self.root.title("BasuraFix · Clasificador de basura con IA")
        self.root.geometry("1140x780")
        self.root.minsize(1000, 700)
        self.root.configure(bg=BG)
        self._set_icon()

        self._init_fonts()
        self._build_ui()
        self._bind_keys()
        self._start_model_load()

        if screenshot:
            self.root.after(11000, lambda: self._capture_and_close(screenshot))

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
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

    def _set_icon(self) -> None:
        icon = Path(__file__).resolve().parents[2] / "assets" / "basurafix.ico"
        if icon.exists():
            try:
                self.root.iconbitmap(str(icon))
            except Exception:
                pass

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=(4, 6))
        body.grid_columnconfigure(0, weight=0, minsize=500)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(
            body, bg=CARD, highlightbackground=BORDER, highlightthickness=1
        )
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
            text="Clasificador inteligente de residuos · CNN entrenada con 3 datasets reales",
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
            text="8 CLASES",
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
        tk.Label(
            head, text="VISTA PREVIA", bg=CARD, fg=MUTED, font=self.f_section
        ).grid(row=0, column=0, sticky="w")
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
            width=470,
            height=390,
        )
        self.preview.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))
        self.preview.bind("<Configure>", lambda event: self._render_preview())

        controls = tk.Frame(parent, bg=CARD)
        controls.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        controls.grid_columnconfigure(0, weight=1)
        self.select_btn = _make_button(
            controls,
            "Seleccionar imagen",
            self.select_file,
            bg=ACCENT,
            fg=ACCENT_DARK,
            hover="#6ee7b7",
            font=self.f_button,
        )
        self.select_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.clear_btn = _make_button(
            controls,
            "Quitar",
            self.clear,
            bg=CARD_2,
            fg=TEXT,
            hover="#233854",
            font=self.f_button,
        )
        self.clear_btn.grid(row=0, column=1, padx=(0, 8))
        self.heat_btn = _make_button(
            controls,
            "Mapa de calor",
            self.show_heatmap,
            bg=CARD_2,
            fg=TEXT,
            hover="#233854",
            font=self.f_button,
        )
        self.heat_btn.grid(row=0, column=2)
        self.heat_btn.configure(state="disabled")

        hint = "Ctrl+O para abrir · también puedes arrastrar y soltar una foto"
        try:
            import tkinterdnd2  # noqa: F401

            hint = "Ctrl+O para abrir · arrastra una foto y suéltala aquí"
        except Exception:
            pass
        tk.Label(
            parent,
            text=hint,
            bg=CARD,
            fg="#6f86a5",
            font=self.f_small,
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))

        # Drag & drop opcional
        self._dnd_enabled = False
        try:
            from tkinterdnd2 import DND_FILES

            self.preview.drop_target_register(DND_FILES)
            self.preview.dnd_bind("<<Drop>>", self._on_drop)
            self._dnd_enabled = True
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
            text="Selecciona una imagen para comenzar",
            bg=CARD_2,
            fg=MUTED,
            font=self.f_body_bold,
            padx=12,
            pady=7,
            anchor="w",
        )
        self.container_label.grid(row=3, column=0, sticky="ew", padx=22)

        self.tip_var = tk.StringVar(
            value="La CNN analizará la foto y te dirá qué residuo es y cómo separarlo."
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
            text="PRINCIPALES COINCIDENCIAS",
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
        tk.Label(
            footer, textvariable=self.status_var, bg=BG, fg=MUTED, font=self.f_small
        ).grid(row=0, column=1, sticky="w")

        self.model_info_var = tk.StringVar(value="")
        tk.Label(
            footer,
            textvariable=self.model_info_var,
            bg=BG,
            fg="#58708f",
            font=self.f_small,
        ).grid(row=0, column=2, sticky="e")

    # ------------------------------------------------------------------
    # Estado y modelo
    # ------------------------------------------------------------------
    def _set_status(self, text: str, kind: str = "info") -> None:
        colors = {"info": MUTED, "ready": ACCENT, "busy": AMBER, "error": RED}
        self.status_var.set(text)
        self.status_dot.itemconfigure(self._status_id, fill=colors.get(kind, MUTED))
        self._status_kind = kind

    def _start_model_load(self) -> None:
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self) -> None:
        started = time.perf_counter()
        try:
            predictor = BasuraPredictor(DEFAULT_MODEL_PATH)
            info = predictor.info()
            elapsed = time.perf_counter() - started
            self.root.after(0, lambda: self._on_model_loaded(predictor, info, elapsed))
        except Exception as exc:  # noqa: BLE001
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
            parts.append(f"Precisión test {accuracy * 100:.1f}%")
        if total:
            parts.append(f"{total:,} imágenes")
        parts.append(f"cargado en {elapsed:.1f} s")
        self.model_info_var.set(" · ".join(parts))
        self._set_status("Modelo listo. Selecciona o arrastra una foto.", "ready")
        self.select_btn.configure(state="normal")
        if self.demo_path:
            self.root.after(300, lambda: self.load_path(self.demo_path))

    def _on_model_error(self, exc: Exception) -> None:
        self._model_ready = False
        self._set_status("No se pudo cargar el modelo", "error")
        self.model_badge.configure(text="MODELO NO DISPONIBLE", fg=RED, bg="#3b1218")
        self.tip_var.set(
            "No se encontró el modelo entrenado. Ejecuta 3_ENTRENAR.bat o "
            "python -m basurafix train, y vuelve a abrir la interfaz."
        )
        self.model_info_var.set(str(exc)[:90])

    # ------------------------------------------------------------------
    # Selección y análisis
    # ------------------------------------------------------------------
    def select_file(self) -> None:
        if not self._model_ready:
            messagebox.showwarning(
                "Modelo no disponible",
                "El modelo aún no está listo. Si es la primera vez, ejecuta 3_ENTRENAR.bat.",
                parent=self.root,
            )
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Selecciona una foto de basura",
            initialdir=self._last_dir,
            filetypes=IMAGE_TYPES,
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        if not self._model_ready:
            self._pending_path = path
            return
        try:
            image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                "No se pudo abrir la imagen",
                f"El archivo no es una imagen válida o está dañado.\n\n{exc}",
                parent=self.root,
            )
            return
        self._image = image
        self._image_path = str(path)
        self._last_dir = str(Path(path).parent)
        self.path_var.set(_shorten(Path(path).name, 52))
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
            self.root.after(0, lambda: self._show_result(result))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self._show_analysis_error(exc))

    def _show_result(self, result: dict) -> None:
        self._result = result
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
        advice = result.get("advice") or ""
        self.tip_var.set(advice)
        self.time_var.set(f"{result['elapsed_ms']:.0f} ms · TTA")
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

    def clear(self) -> None:
        if self._busy:
            return
        self._image = None
        self._image_path = None
        self._result = None
        self.path_var.set("Ninguna imagen seleccionada")
        self.name_var.set("Sin analizar")
        self.conf_var.set("—")
        self.conf_label.configure(fg=MUTED)
        self.name_label.configure(fg=TEXT)
        self.dot.itemconfigure(self._dot_id, fill="#334155")
        self.container_label.configure(
            text="Selecciona una imagen para comenzar", bg=CARD_2, fg=MUTED
        )
        self.tip_var.set(
            "La CNN analizará la foto y te dirá qué residuo es y cómo separarlo."
        )
        self.warning_label.grid_remove()
        self.time_var.set("")
        self._top_probs = [0.0, 0.0, 0.0]
        self._top_colors = ["#334155"] * 3
        for row in self.top_rows:
            row["name"].configure(text="—", fg=MUTED)
            row["pct"].configure(text="—", fg=MUTED)
        self._draw_confidence()
        for index in range(3):
            self._draw_top_bar(index)
        self.heat_btn.configure(state="disabled")
        self._render_preview()
        self._set_status("Selecciona o arrastra una foto para analizarla.", "ready")

    # ------------------------------------------------------------------
    # Vista previa y barras
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
                text="Arrastra una foto aquí",
                fill="#7c93b3",
                font=(self.f_caption.cget("family"), 14, "bold"),
            )
            canvas.create_text(
                width / 2,
                height / 2 + 6,
                text="o pulsa «Seleccionar imagen»",
                fill="#5f7798",
                font=self.f_body,
            )
            canvas.create_text(
                width / 2,
                height / 2 + 42,
                text="JPG · PNG · WEBP · AVIF · HEIC",
                fill="#3f5777",
                font=self.f_small,
            )
            return

        margin = 16
        box = (max(10, width - margin * 2), max(10, height - margin * 2))
        preview = self._image.copy()
        preview.thumbnail(box, Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        canvas.create_image(width / 2, height / 2, image=self._preview_photo)
        canvas.create_rectangle(
            width / 2 - preview.width / 2,
            height / 2 - preview.height / 2,
            width / 2 + preview.width / 2,
            height / 2 + preview.height / 2,
            outline="#2b4366",
            width=1,
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
    # Grad-CAM
    # ------------------------------------------------------------------
    def show_heatmap(self) -> None:
        if self._image is None or self.predictor is None or self._busy:
            return
        self._set_busy(True, "Generando mapa de calor (Grad-CAM)…")
        threading.Thread(target=self._heatmap_worker, args=(self._image,), daemon=True).start()

    def _heatmap_worker(self, image: Image.Image) -> None:
        try:
            explanation = self.predictor.explain(image, size=470)
            self.root.after(0, lambda: self._show_heatmap_window(explanation))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self._show_analysis_error(exc))

    def _show_heatmap_window(self, explanation: dict) -> None:
        self._set_busy(False)
        window = tk.Toplevel(self.root)
        window.title("BasuraFix · Mapa de calor")
        window.configure(bg=BG)
        window.geometry("1010x610")
        window.minsize(900, 540)
        window.transient(self.root)
        window.bind("<Escape>", lambda event: window.destroy())

        header = tk.Frame(window, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 8))
        tk.Label(
            header,
            text="¿Dónde mira la CNN?",
            bg=BG,
            fg=TEXT,
            font=self.f_title,
        ).pack(anchor="w")
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

        labels = ["Foto original", "Mapa de calor (Grad-CAM)"]
        self._heat_photos = []
        for column, (caption, key) in enumerate(
            zip(labels, ("original", "heatmap"))
        ):
            tk.Label(
                body,
                text=caption,
                bg=BG,
                fg=MUTED,
                font=self.f_body_bold,
            ).grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 12, 0))
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
            label = tk.Label(frame, image=photo, bg=CARD)
            label.pack(expand=True, padx=6, pady=6)

        tk.Label(
            window,
            text="El mapa resalta las regiones que más aportaron a la decisión. Úsalo como apoyo visual, no como verdad absoluta.",
            bg=BG,
            fg="#6f86a5",
            font=self.f_small,
            wraplength=900,
            justify="left",
        ).pack(fill="x", padx=24, pady=(10, 14))

    # ------------------------------------------------------------------
    # Drag & drop, atajos y captura
    # ------------------------------------------------------------------
    def _on_drop(self, event) -> None:
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        path = paths[0].strip("{}")
        if Path(path).is_file():
            self.load_path(path)

    def _bind_keys(self) -> None:
        self.root.bind("<Control-o>", lambda event: self.select_file())
        self.root.bind("<Control-O>", lambda event: self.select_file())
        self.root.bind("<Escape>", lambda event: self.clear())
        self.root.bind("<F5>", lambda event: self._analyze())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        try:
            if self.predictor and self.predictor._gradcam:
                self.predictor._gradcam.close()
        except Exception:
            pass
        self.root.destroy()

    def _capture_and_close(self, path: str, attempts: int = 0) -> None:
        if (self._result is None or not self._model_ready) and attempts < 30:
            self.root.after(500, lambda: self._capture_and_close(path, attempts + 1))
            return
        try:
            from PIL import ImageGrab

            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
            self.root.update()
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            print(f"Captura guardada en {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"No se pudo capturar la pantalla: {exc}")
        finally:
            self.root.after(200, self.root.destroy)


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


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Interfaz gráfica de BasuraFix.")
    parser.add_argument("--demo", help="Carga automáticamente una imagen al iniciar.")
    parser.add_argument("--screenshot", help="Guarda una captura de la ventana y cierra.")
    args = parser.parse_args(argv)

    _enable_dpi_awareness()
    try:
        try:
            from tkinterdnd2 import TkinterDnD

            root = TkinterDnD.Tk()
        except Exception:
            root = tk.Tk()
        BasuraFixApp(root, demo_path=args.demo, screenshot=args.screenshot)
        root.mainloop()
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        from datetime import datetime

        from .config import BASE_DIR

        log = BASE_DIR / "reports" / "gui_error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n[{datetime.now().isoformat(timespec='seconds')}]\n")
            fh.write(traceback.format_exc())
        try:
            messagebox.showerror(
                "BasuraFix",
                f"No se pudo iniciar la interfaz:\n\n{exc}\n\nDetalle en {log}",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
