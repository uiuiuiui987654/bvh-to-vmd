@echo off
setlocal
cd /d "%~dp0"

set "PY=E:\NVIDIA Kimodo\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%~dp0kimodo_vmd_converter_gui.py"
