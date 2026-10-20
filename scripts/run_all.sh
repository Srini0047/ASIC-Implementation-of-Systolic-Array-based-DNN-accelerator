#!/usr/bin/env bash
# (needs bash: uses process substitution)
# ONE command for the whole project flow, in the order the docs explain it.
#
#   bash scripts/run_all.sh            # everything
#   bash scripts/run_all.sh --skip-train   # reuse build/model.onnx (faster)
#
# Different network / hardware size (the SAME numbers must go to compiler and RTL):
#   HIDDEN=32 HW_ARGS="--acc-depth 64 --wgt-depth 1024" \
#   HW_G="-GACC_DEPTH=64 -GWGT_DEPTH=1024" bash scripts/run_all.sh
#     HIDDEN : hidden-layer width(s) for sw/train_mlp.py, e.g. HIDDEN=32 or HIDDEN=32,16 (3 layers)
#     HW_ARGS: extra options for sw/compile.py  (--rows, --m-max, --acc-depth, ...)
#     HW_G   : matching Verilator parameter overrides (-GROWS=8 -GCOLS=8 ...)
#
#   step 1  train      PyTorch trains a 64->16->10 digit classifier, exports ONNX
#   step 2  quantize   float weights -> INT8 + INT32 bias + requant (mult,shift)
#   step 3  compile    tile/pack/lay out the network -> register program + C header
#   step 4  simulate   RTL replays the register program (Verilator)
#   step 5  report     accuracy + throughput computed ONLY from RTL readback
#   step 6  driver     C driver register stream == Python stream, replayed on RTL
#   step 7  axi        same program through the AXI4-Lite slave bridge
set -eu
cd "$(dirname "$0")/.."
mkdir -p build
SKIP_TRAIN=0; [ "${1:-}" = "--skip-train" ] && SKIP_TRAIN=1

RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v rtl/axil_reg_bridge.v rtl/dnn_accel_axil_top.v"

if [ $SKIP_TRAIN -eq 0 ]; then
  echo "== step 1: train + export ONNX ==";   python3 sw/train_mlp.py
fi
echo "== step 2: quantize to INT8 ==";         python3 sw/quantize.py
echo "== step 3: compile register program =="; python3 sw/compile.py ${HW_ARGS:-}

echo "== step 4: RTL simulation (Verilator) =="
D=${BUILD_DIR:-/tmp/dnn_sim}
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player ${HW_G:-} \
    --Mdir "$D" tb/tb_dnn_player.v $RTL > "$D.build.log" 2>&1 || { tail -20 "$D.build.log"; exit 1; }
out=$("$D/Vtb_dnn_player" +prog=build/prog.hex +log=build/rtl_readback.txt)
echo "$out" | grep -E "PLAYER"
echo "$out" | grep -q "PLAYER PASSED" || { echo "RTL simulation FAILED"; exit 1; }

echo "== step 5: report (from RTL readback) =="; python3 sw/report.py

echo "== step 6: C driver cross-check =="
bash scripts/check_driver.sh > build/check_driver.log 2>&1 || { tail -20 build/check_driver.log; echo "driver check FAILED"; exit 1; }
grep -E "IDENTICAL|PLAYER|DRIVER CHECK" build/check_driver.log
echo "== step 7: same program through the AXI4-Lite bridge (random stalls) =="
DA=${BUILD_DIR:-/tmp/dnn_sim}_axil
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player \
    -GAXIL=1 ${HW_G:-} --Mdir "$DA" tb/tb_dnn_player.v $RTL > "$DA.build.log" 2>&1 || { tail -20 "$DA.build.log"; exit 1; }
out=$("$DA/Vtb_dnn_player" +prog=build/prog.hex +log=build/axil_readback.txt +axi_stall)
echo "$out" | grep -E "PLAYER"
echo "$out" | grep -q "PLAYER PASSED" || { echo "AXI4-Lite simulation FAILED"; exit 1; }
cmp <(cut -d' ' -f2- build/rtl_readback.txt) <(cut -d' ' -f2- build/axil_readback.txt) \
    && echo "AXI4-Lite readback identical to register-bus readback"
echo "ALL STEPS DONE"
