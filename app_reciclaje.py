"""App con ventana: sube una foto y dice qué tipo de basura es.

Uso:  py app_reciclaje.py
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from usar_modelo import cargar_modelo, predecir_probs

IMG_MOSTRAR = 300  # tamaño de la foto en pantalla


class App:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador de Basura (TrashNet)")
        root.geometry("480x560")
        root.resizable(False, False)

        tk.Label(root, text="CLASIFICADOR DE BASURA",
                 font=("Arial", 16, "bold")).pack(pady=10)
        tk.Label(root, text="Sube una foto y la red neuronal dirá qué es",
                 font=("Arial", 10)).pack()

        tk.Button(root, text="SUBIR IMAGEN", font=("Arial", 12, "bold"),
                  bg="#4CAF50", fg="white", width=20, height=2,
                  command=self.subir).pack(pady=10)

        self.img_label = tk.Label(root, text="(aquí se verá tu foto)",
                                  width=40, height=12, relief="groove")
        self.img_label.pack(pady=5)

        self.resultado = tk.Label(root, text="", font=("Arial", 20, "bold"),
                                  fg="#2E7D32")
        self.resultado.pack(pady=5)

        self.detalle = tk.Label(root, text="", font=("Arial", 11),
                                justify="left")
        self.detalle.pack(pady=5)

        estado = tk.Label(root, text="Cargando modelo...", font=("Arial", 9))
        estado.pack(side="bottom", pady=5)
        root.update()
        self.modelo, self.clases = cargar_modelo("best_trashnet_notebook.pt")
        estado.config(text="Modelo listo (precisión 91.4% en test)")

    def subir(self):
        ruta = filedialog.askopenfilename(
            title="Elige una foto",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not ruta:
            return
        try:
            foto = Image.open(ruta).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir:\n{e}")
            return
        # Miniatura en pantalla
        mini = foto.copy()
        mini.thumbnail((IMG_MOSTRAR, IMG_MOSTRAR))
        self.foto_tk = ImageTk.PhotoImage(mini)  # guardar referencia
        self.img_label.config(image=self.foto_tk, text="")
        # Predecir
        clase, probs = predecir_probs(self.modelo, self.clases, foto)
        self.resultado.config(text=f"Es: {clase.upper()}")
        top = sorted(zip(self.clases, probs), key=lambda t: -t[1])[:3]
        self.detalle.config(
            text="\n".join(f"{c:10s} {p:.1%}" for c, p in top))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
