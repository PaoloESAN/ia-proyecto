@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONPATH=%CD%\src
if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERROR] Ejecuta primero 1_INSTALAR.bat
    pause
    exit /b 1
)
if not exist "models\basurafix_mobilenetv3.pt" (
    echo [ERROR] No existe el modelo entrenado.
    echo Ejecuta 3_ENTRENAR.bat primero.
    pause
    exit /b 1
)
start "" "%CD%\.venv\Scripts\pythonw.exe" -m basurafix gui
