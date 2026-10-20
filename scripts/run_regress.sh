#!/usr/bin/env bash
# Multi-configuration regression. Run from anywhere:
#   bash scripts/run_regress.sh
#
# Primary engine : Verilator --binary --timing   (fast, handles 16x16 easily)
# Cross-check    : Icarus Verilog on the small configs (iverilog re-evaluates
#                  wide flattened buses on every slice update, so it is slow
#                  for arrays larger than ~6x6 -- that is a simulator artefact).
#
# Config columns: ROWS COLS ACT_DEPTH WGT_DEPTH OUT_DEPTH LRB_DEPTH
set -u
cd "$(dirname "$0")/.."
TB=tb_systolic_accel
BUILD=${BUILD_DIR:-/tmp/sa_regress}
mkdir -p "$BUILD"
fail=0

CONFIGS="
4 4 16 16 32 8
8 8 32 32 64 16
4 4 16 16 32 1
4 4 16 16 32 32
3 5 12 12 36 5
6 2 20 24 40 7
1 1 8 4 16 3
2 8 10 8 20 4
16 16 8 32 16 4
"

report() {   # engine tag out
  if echo "$3" | grep -q "TB PASSED"; then
    echo "PASS  [$1] $2  ($(echo "$3" | grep -o 'tests run: [0-9]*'), $(echo "$3" | grep -o 'utilization: [0-9.]* %'))"
  else
    echo "FAIL  [$1] $2"
    echo "$3" | grep -E "FAIL|MISMATCH|ERROR" | head -8
    fail=1
  fi
}

while read -r R C AD WD OD LD; do
  [ -z "${R:-}" ] && continue
  tag="${R}x${C} ACT=$AD WGT=$WD OUT=$OD LRB=$LD"
  d="$BUILD/vl_${R}x${C}_${AD}_${WD}_${OD}_${LD}"

  # ---- Verilator ----
  if verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH \
       --top-module $TB -GROWS=$R -GCOLS=$C -GACT_DEPTH=$AD -GWGT_DEPTH=$WD \
       -GOUT_DEPTH=$OD -GLRB_DEPTH=$LD --Mdir "$d" tb/$TB.v rtl/*.v \
       > "$d.build.log" 2>&1; then
    report verilator "$tag" "$("$d/V$TB" 2>&1)"
  else
    echo "FAIL  [verilator] $tag (build error, see $d.build.log)"; fail=1
  fi

  # ---- Icarus cross-check on small arrays only ----
  if [ $((R*C)) -le 24 ]; then
    sim="$BUILD/iv_${R}x${C}_${AD}_${WD}_${OD}_${LD}.vvp"
    if iverilog -g2005 -o "$sim" \
         -P$TB.ROWS=$R -P$TB.COLS=$C -P$TB.ACT_DEPTH=$AD -P$TB.WGT_DEPTH=$WD \
         -P$TB.OUT_DEPTH=$OD -P$TB.LRB_DEPTH=$LD tb/$TB.v rtl/*.v; then
      report iverilog "$tag" "$(vvp "$sim" 2>&1)"
    else
      echo "FAIL  [iverilog] $tag (compile error)"; fail=1
    fi
  fi
done <<< "$CONFIGS"

[ $fail -eq 0 ] && echo "ALL CONFIGS PASSED" || echo "REGRESSION FAILED"
exit $fail
