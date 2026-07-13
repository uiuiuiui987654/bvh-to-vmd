@echo off
setlocal
cd /d "%~dp0"

set "PY=python"

"%PY%" "%~dp0kimodo_vmd_converter_gui.py"
