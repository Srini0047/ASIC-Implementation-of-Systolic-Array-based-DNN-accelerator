#!/usr/bin/env bash
# DNN engine regression: random multi-layer networks on several hardware
# configurations, each replayed through tb_dnn_player on Verilator.
#   bash scripts/run_dnn_regress.sh            (from anywhere)
# Config columns: ROWS M_MAX KT_MAX NT_MAX ACT_DEPTH WGT_DEPTH BIAS_DEPTH ACC_DEPTH LRB_DEPTH NUM_LAYERS NTESTS
set -u
cd "$(dirname "$0")/.."
BUILD=${BUILD_DIR:-/tmp/dnn_regress}; mkdir -p "$BUILD" build
RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v rtl/axil_reg_bridge.v rtl/dnn_accel_axil_top.v"
fail=0

CONFIGS="
4 8 16 8 256 512 16 32 8 4 40
2 6 12 6 256 512 16 36 4 3 30
8 8 6 6 128 512 16 48 16 4 30
4 5 3 7 64 256 16 24 1 3 40
3 7 8 6 200 256 16 42 5 4 30
1 6 10 5 128 128 16 30 3 3 30
4 8 16 8 256 512 16 32 3 4 40
"

while read -r R M KT NT AD WD BD CD LD NL NT_TESTS; do
  [ -z "${R:-}" ] && continue
  tag="rows=$R M<=$M KT<=$KT NT<=$NT LRB=$LD"
  d="$BUILD/r${R}_${M}_${KT}_${NT}_${AD}_${WD}_${CD}_${LD}"
  prog="$BUILD/prog_$(basename $d).hex"
  python3 sw/gen_random_net.py --rows $R --ntests $NT_TESTS --seed $((R*100+LD)) --out "$prog" \
      --m-max $M --kt-max $KT --nt-max $NT --act-depth $AD --wgt-depth $WD \
      --bias-depth $BD --acc-depth $CD --lrb-depth $LD --num-layers $NL > "$d.gen.log" 2>&1 \
      || { echo "FAIL  [$tag] generator error"; cat "$d.gen.log" | tail -3; fail=1; continue; }
  verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player \
      -GROWS=$R -GCOLS=$R -GM_MAX=$M -GKT_MAX=$KT -GNT_MAX=$NT -GACT_DEPTH=$AD -GWGT_DEPTH=$WD \
      -GBIAS_DEPTH=$BD -GACC_DEPTH=$CD -GLRB_DEPTH=$LD -GNUM_LAYERS=$NL \
      --Mdir "$d" tb/tb_dnn_player.v $RTL > "$d.build.log" 2>&1 \
      || { echo "FAIL  [$tag] build error (see $d.build.log)"; fail=1; continue; }
  out=$("$d/Vtb_dnn_player" +prog="$prog" +log="$d.log" 2>&1)
  if echo "$out" | grep -q "PLAYER PASSED"; then
    echo "PASS  [$tag]  $(echo "$out" | grep -o 'PLAYER: [0-9]* ops, [0-9]* reads')  | $(tail -1 "$d.gen.log" | grep -o '[0-9]* random networks + [0-9]* negative tests')"
  else
    echo "FAIL  [$tag]"; echo "$out" | grep -E "FAIL|MISMATCH|PLAYER:" | head -6; fail=1
  fi
done <<< "$CONFIGS"

# ---- same random program through the AXI4-Lite bridge, with random handshake stalls ----
d="$BUILD/axil_cfg1"; prog="$BUILD/prog_r4_8_16_8_256_512_32_8.hex"
if verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player \
     -GAXIL=1 --Mdir "$d" tb/tb_dnn_player.v $RTL > "$d.build.log" 2>&1; then
  out=$("$d/Vtb_dnn_player" +prog="$prog" +log="$d.log" +axi_stall 2>&1)
  if echo "$out" | grep -q "PLAYER PASSED"; then
    echo "PASS  [AXI4-Lite bridge, random stalls]  $(echo "$out" | grep -o 'PLAYER: [0-9]* ops, [0-9]* reads')"
  else echo "FAIL  [AXI4-Lite bridge]"; echo "$out" | grep -E "FAIL|MISMATCH|PLAYER:" | head -6; fail=1; fi
else echo "FAIL  [AXI4-Lite bridge] build error"; fail=1; fi

[ $fail -eq 0 ] && echo "ALL DNN CONFIGS PASSED" || echo "DNN REGRESSION FAILED"
exit $fail
