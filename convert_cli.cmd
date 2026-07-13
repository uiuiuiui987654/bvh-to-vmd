@echo off
setlocal
cd /d "%~dp0"

if "%~3"=="" (
    echo Usage: %~nx0 "input.bvh" "model.pmx" "output.vmd"
    pause
    exit /b 2
)

set "PY=python"
set "BVH=%~1"
set "PMX=%~2"
set "OUT=%~3"

"%PY%" "%~dp0kimodo_to_mmd_solver.py" --bvh "%BVH%" --pmx "%PMX%" --out "%OUT%" --position-scale auto --motion-fidelity preserve --foot-ik-mode auto --foot-rotation-mode follow-body --wrist-strength 0.45 --finger-mode omit --body-rotation-mode auto --body-frame-mode full --body-rotation-transform normal --knee-hinge flip --leg-solver-mode ccd --pose-solver-mode position --local-rot-feet omit
pause
