@echo off
setlocal
cd /d "%~dp0"

if "%~3"=="" (
    echo Usage: %~nx0 "input.fbx" "model.pmx" "output.vmd"
    pause
    exit /b 2
)

set "PY=python"
set "FBX=%~1"
set "PMX=%~2"
set "OUT=%~3"

"%PY%" "%~dp0fbx_to_vmd.py" --fbx "%FBX%" --pmx "%PMX%" --out "%OUT%"
pause
