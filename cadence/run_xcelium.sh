#!/bin/bash
# Run the accelerator simulation in Cadence Xcelium (xrun).
#
#   bash cadence/run_xcelium.sh                 normal run: the 45-batch digit test, prints PASSED/FAILED
#   bash cadence/run_xcelium.sh --wave          first batch only, writes build/dnn.vcd for the waveform viewer
#   bash cadence/run_xcelium.sh --gui           opens SimVision (interactive; you press Run yourself)
#   bash cadence/run_xcelium.sh --axi           same test through the AXI4-Lite bus bridge, random stalls
#   bash cadence/run_xcelium.sh --prog FILE     use another register program (default build/prog.hex)
#   options can be combined, e.g.  --axi --prog build/tiny_prog.hex
#
# Always works from the project root, wherever you call it from. Leftovers (xcelium.d/, xrun.log, xrun.history)
# are temporary files created by xrun; delete them whenever you like.
# NOTE: written without access to Xcelium - if xrun rejects a flag, the message names it; see docs/BEGINNER_GUIDE.html, section 11 ("If something fails").
cd "$(dirname "$0")/.." || exit 1

PROG=build/prog.hex; LOG=build/rtl_readback.txt
EXTRA=(); MODE=batch
while [ $# -gt 0 ]; do
  case "$1" in
    --wave) MODE=wave; PROG=build/wave_prog.hex; LOG=build/wave_readback.txt; EXTRA+=(-define WAVES) ;;
    --gui)  MODE=gui;  PROG=build/wave_prog.hex; LOG=build/wave_readback.txt; EXTRA+=(-gui -access +rwc) ;;
    --axi)  EXTRA+=(-defparam tb_dnn_player.AXIL=1 +axi_stall); [ "$LOG" = build/rtl_readback.txt ] && LOG=build/axil_readback.txt ;;
    --prog) shift; PROG="$1"; LOG=build/custom_readback.txt ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
[ -f "$PROG" ] || { echo "ERROR: $PROG not found. Run from the project folder; see docs/BEGINNER_GUIDE.html, section 6 (Job 1)."; exit 1; }
command -v xrun >/dev/null || { echo "ERROR: 'xrun' not found. Load the Cadence environment first (your lab's 'source ...' / 'module load xcelium' line)."; exit 1; }

# The testbench goes FIRST so its  `timescale 1ns/1ps  also applies to the design files after it.
xrun -clean -access +r -timescale 1ns/1ps \
     tb/tb_dnn_player.v -f cadence/rtl_files.f -top tb_dnn_player \
     +prog="$PROG" +log="$LOG" "${EXTRA[@]}" | tee build/xrun_last.log

echo
if [ "$MODE" = gui ]; then
  echo ">>> SimVision closed. In GUI mode the PASSED/FAILED line is printed by the testbench inside the Console window, not here."; exit 0
fi
if grep -q "PLAYER PASSED" build/xrun_last.log; then
  echo ">>> RESULT: PASSED  (readback written to $LOG)"
else
  echo ">>> RESULT: FAILED or did not finish - read the lines above (search for 'FAIL' or '*E')"; exit 1
fi
