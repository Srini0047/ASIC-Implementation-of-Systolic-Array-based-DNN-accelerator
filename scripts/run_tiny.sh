#!/usr/bin/env bash
# Run the hand-checkable 4x4 example (sw/example_tiny_net.py) on the RTL.
#   bash scripts/run_tiny.sh
set -eu
cd "$(dirname "$0")/.."
python3 sw/example_tiny_net.py
RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v"
D=${BUILD_DIR:-/tmp/dnn_tiny}
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player \
    --Mdir "$D" tb/tb_dnn_player.v $RTL > "$D.build.log" 2>&1 || { tail -20 "$D.build.log"; exit 1; }
"$D/Vtb_dnn_player" +prog=build/tiny_prog.hex +log=build/tiny_readback.txt | grep PLAYER
echo "raw words the RTL returned (address, data; each word = 4 int8 lanes, lane 0 in bits [7:0]):"
cat build/tiny_readback.txt
python3 - <<'PY'
rows = [l.split() for l in open("build/tiny_readback.txt")]
for img, r in enumerate(rows[-2:]):                   # the last two reads are the two output words
    v = int(r[3], 16)
    lanes = [(v >> (8 * j)) & 0xFF for j in range(4)]
    print(f"image {img}: RTL returned word 0x{v:08x} = lanes {['0x%02X' % x for x in lanes]} "
          f"= {[x - 256 if x >= 128 else x for x in lanes]}")
PY
