# Licencias y atribución

## Código de BasuraFix

Código original creado para este proyecto. Puedes usarlo, modificarlo y compartirlo con fines educativos o de investigación.

## TrashNet

- Autoría: Gary Thung y Mindy Yang (Stanford University).
- Repositorio: https://github.com/garythung/trashnet
- Licencia: MIT.
- Uso: 2.527 imágenes de residuos sobre fondo blanco, 6 clases.

## RealWaste

- Autoría: University of Wollongong (Australia).
- Repositorio: https://huggingface.co/datasets/shahzaibvohra/realwaste
- Licencia: CC BY 4.0.
- Uso: 4.752 imágenes de residuos reales en un vertedero, 9 clases.

## Garbage Classification (12 clases)

- Repositorio: https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes
- Licencia: no especificada por el autor del repositorio. Se usa únicamente con fines educativos y de investigación, sin redistribuir las imágenes originales.
- Uso: 15.520 imágenes de 12 clases (vidrio de colores, ropa, calzado, biológicos, pilas, etc.). Se excluyen las pilas y se limita cada clase a 1.200 imágenes para el entrenamiento.
- Si eres autor de este dataset y deseas que se retire o se corrija la atribución, abre un issue en el proyecto.

## Pesos preentrenados

MobileNetV3-Large con pesos ImageNet, distribuidos por torchvision/PyTorch bajo sus licencias correspondientes (BSD-3-Clause para el código; los pesos tienen sus propios términos de uso).

## Imágenes de ejemplo

- `samples/botella_plastico.jpg`: imagen de prueba proporcionada por el usuario del proyecto.
- Las imágenes de `samples/` extraídas de los datasets conservan la licencia de su fuente.

Si redistribuyes el modelo entrenado, incluye esta atribución y las licencias de TrashNet y RealWaste.
