@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ============================================
echo  BasuraFix - Instalacion
echo ============================================
where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro Python. Instala Python 3.14 (probado) o superior.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -m venv .venv
    if errorlevel 1 goto error
)
echo Actualizando pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo Instalando PyTorch para CPU...
".venv\Scripts\python.exe" -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error
echo Instalando el resto de dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Listo. Ahora ejecuta 2_DESCARGAR_DATOS.bat y luego 3_ENTRENAR.bat.
pause
exit /b 0

:error
echo.
echo [ERROR] La instalacion fallo. Revisa el mensaje anterior.
pause
exit /b 1
