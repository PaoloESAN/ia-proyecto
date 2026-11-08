@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONPATH=%CD%\src
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Ejecuta primero 1_INSTALAR.bat
    pause
    exit /b 1
)
if not exist "data\processed\manifest.csv" (
    echo Preparando los datos por primera vez...
    ".venv\Scripts\python.exe" -m basurafix download
    if errorlevel 1 goto error
    ".venv\Scripts\python.exe" -m basurafix prepare
    if errorlevel 1 goto error
)
echo ============================================
echo  BasuraFix - Entrenamiento
echo ============================================
".venv\Scripts\python.exe" -m basurafix train %*
if errorlevel 1 goto error
echo.
echo Entrenamiento completado. Abre ABRIR_INTERFAZ.bat
pause
exit /b 0

:error
echo.
echo [ERROR] El entrenamiento fallo. Revisa el mensaje anterior.
pause
exit /b 1
