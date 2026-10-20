@echo off
REM Windows: rebuild the chip's input program from the trained network.  Double-click, or in cmd:  cadence\python_flow.bat
REM Needs:  pip install numpy onnx onnxruntime
cd /d "%~dp0.."
echo === step 1: quantize (float numbers -> small integers) ===
python sw\quantize.py
if errorlevel 1 goto fail
echo === step 2: compile (integers -> list of register writes = build\prog.hex) ===
python sw\compile.py
if errorlevel 1 goto fail
echo.
echo DONE. Now run the simulator (Xcelium or ModelSim), then run  cadence\python_report.bat
pause
exit /b 0
:fail
echo.
echo Something failed - read the message just above. See the guide, section "If something fails".
pause
exit /b 1
