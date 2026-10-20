#!/usr/bin/env bash
# Mutation testing: a test suite only means something if it goes RED on a broken design.
# This script injects one deliberate bug at a time into a COPY of the design and checks that the
# tests notice. Every bug must be caught ("KILLED"); a bug the tests miss ("SURVIVED") is a hole
# in the tests and makes this script fail.
#
#   bash scripts/run_mutations.sh            (after: python3 sw/compile.py ; ~2-3 minutes)
#
# Four groups (19 mutations):
#   A. datapath / control RTL bugs      -> detected by the default digit-network program (bit-exact)
#   B. error-handling RTL bugs          -> detected by the random program with negative tests
#   C. C-driver bugs                    -> detected by the driver-vs-Python register-stream diff
#   D. AXI4-Lite bridge bugs            -> detected by the AXI run (random stalls)
# Nothing in rtl/ or driver/ is modified: mutations are applied to copies under $BUILD_DIR.
set -u
cd "$(dirname "$0")/.."
W=${BUILD_DIR:-/tmp/dnn_mut}; rm -rf "$W"; mkdir -p "$W" build
RTLF="pe skew_delay systolic_array sa_core sram_sdp lrb perf_counters rst_sync dnn_ppu dnn_ctrl \
      dnn_accel_core dnn_reg_if dnn_accel_top axil_reg_bridge dnn_accel_axil_top"
killed=0; total=0; bad=0

# mutate FILE OLD NEW : fresh copy of rtl/ + driver/ in $W/tree with ONE exact-text replacement
mutate() {
    rm -rf "$W/tree"; mkdir -p "$W/tree"; cp -r rtl driver "$W/tree/"
    python3 - "$W/tree/$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path).read()
if text.count(old) != 1:
    sys.exit(f"mutation pattern occurs {text.count(old)} times (need exactly 1) in {path}: {old!r}")
open(path, "w").write(text.replace(old, new))
PY
    local rc=$?; [ $rc = 0 ] || bad=1; return $rc
}

verdict() {   # verdict LABEL CAUGHT(0/1) DETAIL
    total=$((total + 1))
    if [ "$2" = 1 ]; then killed=$((killed + 1)); printf "  KILLED    %-58s %s\n" "$1" "$3"
    else                                              printf "  SURVIVED  %-58s %s\n" "$1" "$3"; fi
}

# run_rtl LABEL PROGRAM [extra verilator args] [-- extra plusargs]: build mutated RTL, replay PROGRAM
run_rtl() {
    local label=$1 prog=$2; shift 2
    local vargs=() pargs=() sw=0
    for a in "$@"; do if [ "$a" = -- ]; then sw=1; elif [ $sw = 0 ]; then vargs+=("$a"); else pargs+=("$a"); fi; done
    local files=""; for f in $RTLF; do files="$files $W/tree/rtl/$f.v"; done
    local obj="$W/obj_$(echo "$label" | tr -c 'A-Za-z0-9' _)"
    if ! verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style -Wno-WIDTH --top-module tb_dnn_player \
            "${vargs[@]}" --Mdir "$obj" tb/tb_dnn_player.v $files > "$obj.log" 2>&1; then
        verdict "$label" 0 "(mutant did not build - fix the mutation)"; return
    fi
    local out; out=$(timeout 300 "$obj/Vtb_dnn_player" +prog="$prog" +log="$obj.rb" "${pargs[@]}" 2>&1)
    if echo "$out" | grep -q "PLAYER PASSED"; then verdict "$label" 0 "(tests still pass)"
    else verdict "$label" 1 "$(echo "$out" | grep -m1 -oE 'FAIL[^,]*|[0-9]+ mismatches|[0-9]+ timeouts' | head -1)"; fi
}

#--------------------------------------------------------------------------------------------
echo "A. datapath / control RTL (default digit network, $(wc -l < build/prog.hex) register ops)"
mutate rtl/dnn_ppu.v        '(prod_r + rnd) >>> shift'  'prod_r >>> shift' \
    && run_rtl "PPU: rounding term removed" build/prog.hex
mutate rtl/dnn_ppu.v        'sum_r  <= a + b;'          'sum_r  <= a;' \
    && run_rtl "PPU: bias not added" build/prog.hex
mutate rtl/dnn_accel_core.v 'first_kt ? {ACC_WIDTH{1'"'"'b0}}' "1'b0 ? {ACC_WIDTH{1'b0}}" \
    && run_rtl "core: stale partial sum not masked on first K-tile" build/prog.hex
mutate rtl/dnn_ctrl.v       'act_addr <= act_addr + kt_ext;' "act_addr <= act_addr + kt_ext + 1'b1;" \
    && run_rtl "ctrl: activation address stride off by one" build/prog.hex
mutate rtl/dnn_ctrl.v       '{1'"'"'b0, w_ptr} + ROWS_M1 - ld_ext' "{1'b0, w_ptr} + ld_ext" \
    && run_rtl "ctrl: weight rows loaded in the wrong order" build/prog.hex
mutate rtl/sa_core.v        'vfull[LATENCY-1];' 'vfull[LATENCY];' \
    && run_rtl "sa_core: res_vld_early one cycle too LATE" build/prog.hex
mutate rtl/sa_core.v        'vfull[LATENCY-1];' 'vfull[LATENCY-2];' \
    && run_rtl "sa_core: res_vld_early one cycle too EARLY" build/prog.hex

#--------------------------------------------------------------------------------------------
echo "B. error-handling RTL (random program incl. negative tests, default hardware sizes)"
python3 sw/gen_random_net.py --rows 4 --ntests 40 --seed 408 --out "$W/rand.hex" > "$W/rand.gen.log" 2>&1 \
    || { tail -3 "$W/rand.gen.log"; echo "generator failed"; exit 1; }
mutate rtl/dnn_reg_if.v     'err & ~start_r,' 'err,' \
    && run_rtl "reg_if: stale err visible right after START" "$W/rand.hex"
mutate rtl/dnn_reg_if.v     '(reg_wdata > NUM_LAYERS) ? {LAY_W{1'"'"'b0}} : reg_wdata[LAY_W-1:0]' 'reg_wdata[LAY_W-1:0]' \
    && run_rtl "reg_if: N_LAYERS out-of-range value wraps instead of clamping" "$W/rand.hex"
mutate rtl/dnn_ctrl.v       'range_ok && (acc_rows <= ACC_DEPTH)' 'range_ok' \
    && run_rtl "ctrl: M*NT <= ACC_DEPTH check removed" "$W/rand.hex"
mutate rtl/dnn_ctrl.v       '(f_shift <= SHIFT_MAX)' "1'b1" \
    && run_rtl "ctrl: SHIFT range check removed" "$W/rand.hex"

#--------------------------------------------------------------------------------------------
echo "C. C driver (register stream must equal the Python compiler's, byte for byte)"
n_first=$(python3 -c "import json; print(json.load(open('build/prog_meta.json'))['n_first'])")
head -n "$n_first" build/prog.hex > "$W/py_first.hex"
run_drv() {
    local label=$1
    if ! gcc -std=c99 -O1 -w -I"$W/tree/driver" -Ibuild "$W/tree/driver/dnn_accel.c" "$W/tree/driver/example_infer.c" \
            -o "$W/drv" 2> "$W/drv.log"; then verdict "$label" 0 "(mutant did not compile)"; return; fi
    local out; out=$(timeout 60 "$W/drv" "$W/c_trace.hex" 2>&1); local rc=$?
    if [ $rc = 0 ] && cmp -s "$W/py_first.hex" "$W/c_trace.hex" && echo "$out" | grep -q "output unpacking OK"; then
        verdict "$label" 0 "(driver output identical)"
    else verdict "$label" 1 "(stream differs or output check failed)"; fi
}
mutate driver/dnn_accel.c \
'        wr(d, base + DNN_F_SRC,   (uint32_t)d->lay[i].src);
        wr(d, base + DNN_F_DST,   (uint32_t)d->lay[i].dst);' \
'        wr(d, base + DNN_F_SRC,   (uint32_t)d->lay[i].dst);
        wr(d, base + DNN_F_DST,   (uint32_t)d->lay[i].src);' \
    && run_drv "driver: descriptor SRC and DST swapped"
mutate driver/dnn_accel.c 'wbase + p * R + r;' 'wbase + p * R + (R - 1 - r);' \
    && run_drv "driver: weight rows in reversed order"
mutate driver/dnn_accel.c '(word * h->stride_b + c)' '(word * h->stride_b + (R - 1 - c))' \
    && run_drv "driver: bias lanes reversed"
mutate driver/dnn_accel.c 'y[(size_t)m * np + nt * R + lane]' 'y[(size_t)m * np + lane * nt_n + nt]' \
    && run_drv "driver: output read-back stride wrong"

#--------------------------------------------------------------------------------------------
echo "D. AXI4-Lite bridge (default program through AXI, random handshake stalls)"
mutate rtl/axil_reg_bridge.v 'assign s_axil_awready = idle & wr_req;' 'assign s_axil_awready = idle & s_axil_awvalid;' \
    && run_rtl "bridge: AWREADY asserted without WVALID" build/prog.hex -GAXIL=1 -- +axi_stall
mutate rtl/axil_reg_bridge.v 'assign s_axil_wready  = idle & wr_req;' 'assign s_axil_wready  = idle & s_axil_wvalid;' \
    && run_rtl "bridge: WREADY asserted without AWVALID" build/prog.hex -GAXIL=1 -- +axi_stall
mutate rtl/axil_reg_bridge.v 'reg_addr      <= s_axil_awaddr[17:2];' 'reg_addr      <= s_axil_awaddr[16:1];' \
    && run_rtl "bridge: wrong write-address slice" build/prog.hex -GAXIL=1 -- +axi_stall
mutate rtl/axil_reg_bridge.v \
'                        reg_rd_en <= 1'"'"'b1;
                        state     <= S_R0;' \
'                        reg_rd_en <= 1'"'"'b1;
                        s_axil_rvalid <= 1'"'"'b1;
                        state     <= S_R1;' \
    && run_rtl "bridge: RVALID one cycle too early" build/prog.hex -GAXIL=1 -- +axi_stall

echo
echo "mutations killed: $killed of $total"
[ "$bad" = 0 ] || echo "(at least one mutation pattern no longer matches the source - update this script)"
[ "$killed" = "$total" ] && [ "$bad" = 0 ] && { echo "ALL MUTATIONS KILLED"; exit 0; } || { echo "SOME MUTATIONS SURVIVED"; exit 1; }
