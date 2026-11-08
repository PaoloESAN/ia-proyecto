# ♻️ IA Proyecto — BasuraFix (Clasificación de Basura de Máxima Precisión)

Proyecto mejorado integrando el pipeline completo de **BasuraFix** (MobileNetV3-Large entrenada sobre **TrashNet + RealWaste + Garbage Classification 12 clases**, con **96.34% de precisión en test con TTA** y **99.79% Top-3**) junto con los scripts y notebooks originales del repositorio.

![Interfaz de BasuraFix](reports/screenshot.png)

---

## 🚀 Mejoras de Precisión e Ingeniería Integradas

Comparado con el baseline original (ResNet18 solo con TrashNet, ~91.4% test accuracy en 6 clases sobre fondo blanco), la integración de **BasuraFix** aporta:

| Aspecto | Baseline Anterior (`TrashNet`) | Nuevo Pipeline Integrado (`BasuraFix`) |
|---|---|---|
| **Datasets** | 1 fuente (`TrashNet`: 2,527 fotos de estudio) | **3 fuentes unificadas** (`TrashNet` + `RealWaste` + `Garbage12`): **12,762 imágenes únicas** tras deduplicar **2,174 duplicados exactos por SHA-1** y balancear a máx. 1,200 por clase |
| **Clases** | 6 (`cardboard, glass, metal, paper, plastic, trash`) | **8 clases reales** (+ `organic` y `textile`, excluyendo pilas peligrosas) con mapeo a contenedores según **NTP 900.058 (Perú)** |
| **Preprocesamiento** | Resize simple + Flip/Rotación/ColorJitter | Corrección **EXIF**, manejo de transparencia RGBA, `RandomResizedCrop` antialias, `GaussianBlur`, `RandomErasing` y evaluación con `Resize(256) + CenterCrop(224)` |
| **Arquitectura y Regularización** | ResNet18 + CrossEntropy simple + StepLR | **MobileNetV3-Large** (también soporta EfficientNet-B0/B1, ConvNeXt-Tiny, ResNet50) + **Dropout 0.25** + **Label Smoothing (0.10)** + **Pesos de clase por raíz inversa** + **AdamW con LR diferencial** + **Cosine Annealing** + **Grad Clip (5.0)** |
| **Inferencia** | 1 pasada directa | **TTA (Test-Time Augmentation)** promediando imagen original + espejo horizontal, detección de incertidumbre/ambigüedad y **mapas de calor Grad-CAM** |
| **Precisión en Test** | 91.40% (2,527 imgs, 6 clases) | **96.24% (96.34% con TTA)** · **F1 Macro: 0.9626** · **Top-3: 99.79%** (sobre 1,914 imágenes de test reales) |

---

## 🖥️ Uso Rápido

El modelo ya viene entrenado e integrado en `models/basurafix_mobilenetv3.pt`.

### 1. Abrir la Interfaz Gráfica (Tkinter + Drag & Drop + Grad-CAM)

Puedes hacer doble clic en `ABRIR_INTERFAZ.bat` o ejecutar:

```powershell
.\.venv\Scripts\python.exe app_reciclaje.py
# o con una imagen de demostración:
.\.venv\Scripts\python.exe app_reciclaje.py --demo samples/botella_plastico.jpg
```

### 2. Clasificar una Imagen por Consola (con Grad-CAM opcional)

```powershell
.\.venv\Scripts\python.exe usar_modelo.py samples/botella_plastico.jpg
.\.venv\Scripts\python.exe usar_modelo.py samples/botella_plastico.jpg --explain reports/mi_gradcam.png
```

También puedes usar el CLI completo del paquete `basurafix`:

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m basurafix predict samples/botella_plastico.jpg --explain reports/explicacion.png
```

### 3. Reentrenar o Descargar el Dataset Combinado (Opcional)

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m basurafix download   # Descarga TrashNet + RealWaste + Garbage12
.\.venv\Scripts\python.exe -m basurafix prepare    # Deduplica (SHA-1) y crea split estratificado 70/15/15
.\.venv\Scripts\python.exe -m basurafix train      # Entrena en 2 fases y genera reportes en reports/
```

---

## 📊 Resultados por Clase (Test Set — 1,914 imágenes)

| Clase | Precisión | Recall | F1-Score | Soporte | Contenedor (NTP 900.058) |
|---|---:|---:|---:|---:|---|
| **Cartón** (`cardboard`) | 96.57% | 97.04% | 0.9681 | 203 | 🔵 Azul · Papel y cartón |
| **Vidrio** (`glass`) | 96.39% | 97.09% | 0.9674 | 275 | 🟢 Verde · Vidrio |
| **Metal** (`metal`) | 96.52% | 94.87% | 0.9569 | 234 | 🟡 Amarillo · Plásticos y metales |
| **Papel** (`paper`) | 95.36% | 97.41% | 0.9638 | 232 | 🔵 Azul · Papel y cartón |
| **Plástico** (`plastic`) | 96.20% | 94.40% | 0.9529 | 268 | 🟡 Amarillo · Plásticos y metales |
| **Orgánico** (`organic`) | 98.19% | 98.55% | 0.9837 | 275 | 🟤 Marrón · Residuos orgánicos |
| **Textil** (`textile`) | 95.69% | 97.37% | 0.9652 | 228 | 🟣 Punto de acopio textil |
| **Basura general** (`trash`) | 94.36% | 92.46% | 0.9340 | 199 | ⚫ Negro/gris · No reciclable |
| **Total / Promedio Macro** | **96.16%** | **96.15%** | **0.9615** (**0.9626 TTA**) | **1,914** | **Accuracy: 96.24% (96.34% TTA)** |

---

## 📁 Estructura del Proyecto

```
ia-proyecto/
├── app_reciclaje.py                # Lanza la interfaz moderna de BasuraFix
├── usar_modelo.py                  # Inferencia CLI + API compatible (con TTA y Grad-CAM)
├── models/
│   └── basurafix_mobilenetv3.pt    # Modelo MobileNetV3-Large entrenado (96.34% test TTA)
├── reports/                        # Matriz de confusión, curvas, métricas JSON y Grad-CAMs
├── samples/                        # Fotos de prueba reales
├── src/
│   ├── dataset.py                  # Carga de TrashNet HF + split estratificado
│   ├── preprocessing.py            # Preprocesamiento mejorado (EXIF, RandomErasing, Blur)
│   └── basurafix/                  # Paquete completo multi-dataset (download, prepare, train, gui, gradcam)
└── notebooks/
    └── 01_lectura_preprocesamiento.ipynb
```
