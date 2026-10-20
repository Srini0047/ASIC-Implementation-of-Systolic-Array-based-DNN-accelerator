#!/usr/bin/env bash
# Cross-check the C host driver against the Python compiler AND the RTL.
#   bash scripts/check_driver.sh        (after: python3 sw/compile.py)
#
#   1. compile driver + example (strict warnings, -Werror)
#   2. run it: it records every register access to build/c_trace.hex
#   3. the recorded stream must equal the first n_first lines of build/prog.hex
#      (probe + weights + biases + descriptors + batch 0), byte for byte
#   4. replay the C stream on the RTL (Verilator) - it must PASS
set -eu
cd "$(dirname "$0")/.."
mkdir -p build

echo "[1/4] compile C driver"
gcc -std=c99 -Wall -Wextra -Wpedantic -Werror -O2 -Idriver -Ibuild \
    driver/dnn_accel.c driver/example_infer.c -o build/example_infer
gcc -std=c99 -Wall -Wextra -Wpedantic -Werror -fsyntax-only -x c - -Idriver <<'EOF'
#include "dnn_accel.h"
#include "hal_mmio.h"
static dnn_mmio_t m; static dnn_hal_t h = { &m, dnn_mmio_read, dnn_mmio_write, 0 };
int hal_check(dnn_dev_t *d) { return dnn_probe(d, &h); }
EOF

echo "[2/4] run driver (recorder HAL)"
./build/example_infer build/c_trace.hex | tail -3

echo "[3/4] compare with the Python-generated register stream"
n_first=$(python3 -c "import json; print(json.load(open('build/prog_meta.json'))['n_first'])")
head -n "$n_first" build/prog.hex > build/py_trace_first.hex
if cmp -s build/py_trace_first.hex build/c_trace.hex; then
    echo "      IDENTICAL: $n_first register accesses (C driver == Python compiler)"
else
    echo "      DIFFERENT - first differing lines:"; diff build/py_trace_first.hex build/c_trace.hex | head -8; exit 1
fi

echo "[4/4] replay the C stream on the RTL"
{ cat build/c_trace.hex; echo "ff00000000000000"; } > build/c_prog.hex
RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v"
D=${BUILD_DIR:-/tmp/dnn_driver_chk}
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player ${HW_G:-} \
    --Mdir "$D" tb/tb_dnn_player.v $RTL > "$D.build.log" 2>&1 || { tail -20 "$D.build.log"; exit 1; }
out=$("$D/Vtb_dnn_player" +prog=build/c_prog.hex +log=build/c_rtl_readback.txt 2>&1)
echo "$out" | grep -E "PLAYER" | sed 's/^/      /'
echo "$out" | grep -q "PLAYER PASSED" && echo "DRIVER CHECK PASSED" || { echo "DRIVER CHECK FAILED"; exit 1; }
