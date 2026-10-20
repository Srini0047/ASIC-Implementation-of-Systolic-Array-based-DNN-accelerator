@echo off
REM ModelSim on Windows.  Double-click this file, or in a cmd window (any folder) type:
REM    cadence\run_modelsim.bat            normal run
REM    cadence\run_modelsim.bat axi        through the AXI4-Lite bridge
REM    cadence\run_modelsim.bat gui        open the ModelSim window with waveform
REM    cadence\run_modelsim.bat tiny       the 4-number hand-checkable example
REM    cadence\run_modelsim.bat pe         the single-PE testbench (15 checks, console)
REM    cadence\run_modelsim.bat pegui      the single-PE testbench with a waveform window
cd /d "%~dp0.."
where vsim >nul 2>nul
if errorlevel 1 (
  echo ERROR: vsim was not found. Open "ModelSim" from the Start menu once to find its folder,
  echo        or add its win32 or win64 folder to PATH. See docs\BEGINNER_GUIDE.html, section 11, "If something fails".
  pause
  exit /b 1
)
if not exist build\prog.hex (
  echo ERROR: build\prog.hex not found - are you inside the unzipped project?
  pause
  exit /b 1
)
if /I "%1"=="pe" (
  vsim -c -do cadence/run_pe_modelsim.do
  echo.
  echo Look for the line  "*** tb_pe PASSED ***"  above.
  pause
  exit /b 0
)
if /I "%1"=="pegui" (
  vsim -do cadence/run_pe_modelsim_gui.do
  exit /b 0
)
if /I "%1"=="axi" (
  vsim -c -do cadence/run_modelsim_axi.do
) else if /I "%1"=="tiny" (
  vsim -c -do cadence/run_modelsim_tiny.do
) else if /I "%1"=="gui" (
  vsim -do cadence/run_modelsim_gui.do
) else (
  vsim -c -do cadence/run_modelsim.do
)
echo.
echo Look for the line  "*** PLAYER PASSED ***"  above.
pause
