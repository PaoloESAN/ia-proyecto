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
echo ============================================
echo  BasuraFix - Descarga y preparacion de datos
echo ============================================
".venv\Scripts\python.exe" -m basurafix download
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m basurafix prepare
if errorlevel 1 goto error
echo.
echo Datos listos. Ahora ejecuta 3_ENTRENAR.bat
pause
exit /b 0

:error
echo.
echo [ERROR] Revisa el mensaje anterior.
pause
exit /b 1
