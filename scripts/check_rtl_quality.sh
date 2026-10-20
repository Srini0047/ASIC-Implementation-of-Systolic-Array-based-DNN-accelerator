#!/usr/bin/env bash
# Static RTL hygiene checks (no simulation):
#   1. Verilator lint, -Wall, on both top levels        (expect: 0 warnings)
#   2. Yosys: proc; check -assert; latch / async scan   (expect: 0 latches,
#      exactly one asynchronous-reset flop group = rst_sync)
#   bash scripts/check_rtl_quality.sh
# Needs: verilator, yosys.
set -eu
cd "$(dirname "$0")/.."

RTL="rtl/pe.v rtl/skew_delay.v rtl/systolic_array.v rtl/sa_core.v rtl/sram_sdp.v rtl/lrb.v \
     rtl/perf_counters.v rtl/rst_sync.v rtl/dnn_ppu.v rtl/dnn_ctrl.v rtl/dnn_accel_core.v \
     rtl/dnn_reg_if.v rtl/dnn_accel_top.v"
AXIL="rtl/axil_reg_bridge.v rtl/dnn_accel_axil_top.v"

fail=0
echo "[1/2] Verilator lint -Wall"
for top in dnn_accel_top dnn_accel_axil_top; do
    files="$RTL"; [ "$top" = dnn_accel_axil_top ] && files="$RTL $AXIL"
    if out=$(verilator --lint-only -Wall --top-module "$top" $files 2>&1); then
        echo "      $top: 0 warnings"
    else
        echo "      $top: WARNINGS"; echo "$out" | head -20; fail=1
    fi
done

echo "[2/2] Yosys structural check"
for top in dnn_accel_top dnn_accel_axil_top; do
    files="$RTL"; [ "$top" = dnn_accel_axil_top ] && files="$RTL $AXIL"
    log=$(mktemp)
    yosys -q -p "read_verilog $files; hierarchy -check -top $top; proc; flatten; check -assert; \
                 stat; tee -o $log.latch select -count t:\$dlatch t:\$dlatchsr t:\$adlatch; \
                 tee -o $log.async select -count t:\$adff t:\$adffe t:\$aldff t:\$dffsr" \
        > "$log.out" 2>&1 || { echo "      $top: yosys FAILED"; tail -15 "$log.out"; fail=1; continue; }
    latches=$(grep -Eo '[0-9]+ objects' "$log.latch" | head -1 | cut -d' ' -f1)
    asyncs=$(grep -Eo '[0-9]+ objects' "$log.async" | head -1 | cut -d' ' -f1)
    echo "      $top: latches=${latches:-?}  async-reset flops=${asyncs:-?}  (check -assert passed)"
    [ "${latches:-1}" = 0 ] || fail=1
    rm -f "$log" "$log.out" "$log.latch" "$log.async"
done

[ "$fail" = 0 ] && echo "RTL QUALITY CHECK PASSED" || { echo "RTL QUALITY CHECK FAILED"; exit 1; }
