@echo off
setlocal
cd /d "%~dp0"

set "PY=E:\NVIDIA Kimodo\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "BVH=H:\3D\人物\output3.bvh"
set "PMX=H:\3D\人物\鸣潮_千咲.pmx"
set "OUT=H:\3D\人物\output3_kimodo_fixed.vmd"

if not "%~1"=="" set "BVH=%~1"
if not "%~2"=="" set "PMX=%~2"
if not "%~3"=="" set "OUT=%~3"

"%PY%" "%~dp0kimodo_to_mmd_solver.py" --bvh "%BVH%" --pmx "%PMX%" --out "%OUT%" --foot-ik-mode auto --wrist-strength 0.45 --hand-outward 0.9 --hand-forward -0.25 --hand-down 2.0 --finger-mode omit --body-rotation-mode auto
pause
