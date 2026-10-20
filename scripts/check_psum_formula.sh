#!/usr/bin/env bash
# Check the partial-sum traffic formula used by sw/report.py and sw/sweep_arch.py
# against the RTL itself:
#     per layer, per batch:   (KT-1) * M * NT   accumulator-SRAM writes  (and the same number of reads)
# The testbench (built with -DSTROBES) counts every acc_wr_en / acc_rd_en strobe of the whole run.
#   bash scripts/check_psum_formula.sh        (after: python3 sw/compile.py)
set -eu
cd "$(dirname "$0")/.."
RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v"
D=${BUILD_DIR:-/tmp/dnn_strobes}
verilator --binary --timing -DSTROBES -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH ${HW_G:-} \
    --top-module tb_dnn_player --Mdir "$D" tb/tb_dnn_player.v $RTL > "$D.build.log" 2>&1 \
    || { tail -20 "$D.build.log"; exit 1; }
out=$("$D/Vtb_dnn_player" +prog=build/prog.hex +log=build/strobes_readback.txt 2>&1)
echo "$out" | grep -q "PLAYER PASSED" || { echo "$out" | tail -5; echo "RTL run failed"; exit 1; }
python3 - "$out" <<'PY'
import json, re, sys
meta = json.load(open("build/prog_meta.json"))
m = re.search(r"STROBES acc_wr_en=(\d+) acc_rd_en=(\d+)", sys.argv[1])
wr, rd = int(m.group(1)), int(m.group(2))
per_batch = sum((L["KT"] - 1) * meta["batch"] * L["NT"] for L in meta["layers"])
expect = per_batch * meta["nbatches"]
print(f"formula : {per_batch} words/batch x {meta['nbatches']} batches = {expect}")
print(f"RTL     : acc_wr_en strobes = {wr}, acc_rd_en strobes = {rd}")
ok = (wr == expect and rd == expect)
print("PSUM FORMULA CHECK PASSED" if ok else "PSUM FORMULA CHECK FAILED")
sys.exit(0 if ok else 1)
PY
