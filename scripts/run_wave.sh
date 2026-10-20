#!/usr/bin/env bash
# Dump a waveform (build/dnn.vcd) of the FIRST batch of the digit network:
# probe + weight/bias/descriptor load + 8 images through both layers + readback.
#   bash scripts/run_wave.sh            (after python3 sw/compile.py)
#   gtkwave build/dnn.vcd               (or Simvision / Verdi / Surfer)
# Signals worth adding first (hierarchy tb_dnn_player.g_reg.dut.*):
#   u_core.u_ctrl.state  (0 IDLE 1 FETCH 2 LDW 3 COMP 4 DRAIN)   u_core.u_ctrl.layer_cnt/kt_cnt/nt_cnt/m_cnt
#   u_core.u_core.*      (sa_core: skewed inputs, PE psums, res_vld)   u_core.acc_wr_en / acc_rd_en
#   u_core.u_ppu.*       (requantization pipeline)                u_if.reg_* (the register bus)
set -eu
cd "$(dirname "$0")/.."
mkdir -p build
n_first=$(python3 -c "import json; print(json.load(open('build/prog_meta.json'))['n_first'])")
{ head -n "$n_first" build/prog.hex; echo "ff00000000000000"; } > build/wave_prog.hex
RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v"
D=${BUILD_DIR:-/tmp/dnn_wave}
verilator --binary --timing --trace -DWAVES -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH \
    --top-module tb_dnn_player --Mdir "$D" tb/tb_dnn_player.v $RTL > "$D.build.log" 2>&1 \
    || { tail -20 "$D.build.log"; exit 1; }
"$D/Vtb_dnn_player" +prog=build/wave_prog.hex +log=build/wave_readback.txt | grep PLAYER
ls -la build/dnn.vcd
