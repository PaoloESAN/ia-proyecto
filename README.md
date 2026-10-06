# IA Proyecto — Clasificación de Basura (TrashNet)

Proyecto basado en [garythung/trashnet](https://github.com/garythung/trashnet)
(Stanford CS 229). Dataset espejo en HuggingFace: `kuchidareo/small_trashnet`
(2527 fotos, 6 clases: glass, paper, cardboard, plastic, metal, trash).

## Ramas

- `main`: solo Experimento 1 — lectura del dataset y preprocesamiento.
- `dev`: todo lo avanzado hasta ahora (entrenamiento, modelo `.pt`, app).

## Experimento 1 (main)

Lectura del dataset + preprocesamiento:
split 70/13/17 train/val/test + transforms + DataLoaders.

## Cómo correr (main)

Con uv (recomendado):

```bash
uv sync
# abrir notebooks/01_lectura_preprocesamiento.ipynb en VS Code / Jupyter
```

Con pip (alternativa):

```bash
py -m pip install -r requirements.txt
```

## Estructura (main)

```
README.md
pyproject.toml + uv.lock + .python-version  # entorno uv
requirements.txt                            # alternativa pip (se conserva)
notebooks/01_lectura_preprocesamiento.ipynb  # exp.1 autocontenido: lectura + prepro
```
