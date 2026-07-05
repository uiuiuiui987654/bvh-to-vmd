@echo off
setlocal
cd /d "%~dp0"

set "PY=E:\NVIDIA Kimodo\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -m PyInstaller --noconfirm --windowed --name KimodoVMDConverter --add-data "kimodo_to_mmd_solver.py;." --add-data "render_mmd_pose.py;." kimodo_vmd_converter_gui.py
pause
