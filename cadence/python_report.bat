@echo off
REM Windows: after the simulator has written build\rtl_readback.txt, turn it into the accuracy / speed report.
cd /d "%~dp0.."
if not exist build\rtl_readback.txt (
  echo build\rtl_readback.txt is missing - run the simulator first.
  pause
  exit /b 1
)
python sw\report.py
pause
