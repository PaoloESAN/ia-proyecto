"""App gráfica BasuraFix: sube o arrastra una foto y dice qué residuo es,
en qué contenedor depositarlo y muestra su mapa de calor Grad-CAM.

Entrenado con 12,762 imágenes reales (TrashNet + RealWaste + Garbage12),
MobileNetV3-Large + TTA (96.34% de precisión en test).

Uso:
  py app_reciclaje.py
  py app_reciclaje.py --demo samples/botella_plastico.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from basurafix.gui import BasuraFixApp as App, main  # noqa: E402, F401

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
