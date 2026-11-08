"""Punto de entrada de BasuraFix.

    python -m basurafix                 -> abre la interfaz
    python -m basurafix download        -> descarga TrashNet + RealWaste
    python -m basurafix prepare         -> unifica y divide los datos
    python -m basurafix train           -> entrena la CNN
    python -m basurafix gui             -> abre la interfaz
    python -m basurafix predict FOTO    -> clasifica una imagen por consola
"""

from __future__ import annotations

import sys

from .utils import setup_console

COMMANDS = {
    "download": "Descarga TrashNet y RealWaste.",
    "prepare": "Unifica los datasets y crea train/val/test.",
    "train": "Entrena la CNN con transfer learning.",
    "gui": "Abre la interfaz gráfica.",
    "predict": "Clasifica una imagen desde la consola.",
}


def _help() -> None:
    print("BasuraFix · clasificador de residuos con CNN\n")
    print("Uso: python -m basurafix <comando> [opciones]\n")
    print("Comandos:")
    for name, description in COMMANDS.items():
        print(f"  {name:<10} {description}")
    print("\nSin comando se abre la interfaz gráfica.")
    print("Ejemplo: python -m basurafix predict samples/botella_plastico.jpg")


def main() -> int:
    setup_console()
    if len(sys.argv) < 2:
        from .gui import main as gui_main

        return gui_main([])

    command = sys.argv[1].lower()
    if command in {"-h", "--help", "help"}:
        _help()
        return 0
    if command not in COMMANDS:
        print(f"Comando desconocido: {command}\n", file=sys.stderr)
        _help()
        return 1

    if command == "download":
        from .download_data import main as run
    elif command == "prepare":
        from .prepare_data import main as run
    elif command == "train":
        from .train import main as run
    elif command == "gui":
        from .gui import main as run
    else:
        from .predict import main as run
    return run(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
