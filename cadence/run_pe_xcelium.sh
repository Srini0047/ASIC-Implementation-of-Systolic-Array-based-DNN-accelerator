#!/bin/bash
# Run the single-PE self-checking testbench (tb/tb_pe.v) in Cadence Xcelium.
#
#   bash cadence/run_pe_xcelium.sh          console run: prints 15 "[ok]" lines and "*** tb_pe PASSED ***"
#   bash cadence/run_pe_xcelium.sh --gui    opens SimVision so you can look at the waveform (press Run yourself)
#
# Works from the project root wherever you call it from. xrun leaves xcelium.d/, xrun.log, xrun.history behind;
# they are temporary and you can delete them.
# NOTE: written without access to Xcelium - if xrun rejects a flag, the message names it.
cd "$(dirname "$0")/.." || exit 1
mkdir -p build
EXTRA=()
GUI=0
if [ "$1" = "--gui" ]; then GUI=1; EXTRA=(-gui -access +rwc); fi
command -v xrun >/dev/null || { echo "ERROR: 'xrun' not found. Load the Cadence environment first (your lab's 'source ...' / 'module load xcelium' line)."; exit 1; }

# The testbench goes FIRST so its  `timescale  is the one that is active when pe.v is compiled.
xrun -clean -access +r -timescale 1ns/1ps tb/tb_pe.v rtl/pe.v -top tb_pe "${EXTRA[@]}" | tee build/xrun_pe_last.log

echo
if [ "$GUI" = 1 ]; then
  echo ">>> SimVision closed. In GUI mode the PASSED/FAILED line is printed inside the Console window."; exit 0
fi
if grep -q "tb_pe PASSED" build/xrun_pe_last.log; then
  echo ">>> RESULT: PASSED"
else
  echo ">>> RESULT: FAILED or did not finish - read the lines above (search for '[FAIL]' or '*E')"; exit 1
fi
