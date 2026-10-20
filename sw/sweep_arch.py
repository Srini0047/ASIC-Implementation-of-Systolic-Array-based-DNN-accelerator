#!/usr/bin/env python3
"""Architecture sweep: how do array size and batch size change speed and data movement?

  python3 sw/sweep_arch.py                 # default 5 configurations
  python3 sw/sweep_arch.py --rows 8 --batch 16   # one custom point

For every configuration this script
  1. compiles the SAME quantized digit network for that hardware size (sw/compile.py),
  2. builds the RTL with those parameters (Verilator) and runs it (tb_dnn_player),
  3. reads the RTL's own performance counters back and prints a comparison.

Everything in the table comes from running the RTL, except the "psum" column
(partial-sum traffic), which is computed from the tiling geometry - the RTL has no
performance counter for it. The formula is  (KT-1) * M * NT  writes and the same number
of reads per layer per batch, each moving ROWS*4 bytes. The testbench is built with
-DSTROBES, which counts the real acc_wr_en / acc_rd_en strobes, and this script fails
if they differ from the formula (so the check is repeated for every configuration).

Each configuration is a fresh hardware instance sized for the batch:
LRB_DEPTH = batch, ACC_DEPTH = batch*NT, ACT_DEPTH = 2*batch*KT (see hw_for). Only
3 batches (24 or 96 images) are simulated per point; the per-batch counters are read
from the last batch. Cycle counts do not depend on the pixel values, so this is enough.

What to look for (it is the whole point of the exercise):
  * cycles/image falls with a bigger array, but MAC utilization falls too, because
    this network is tiny (64->16->10): a 16x16 array is mostly idle.
  * bytes moved per useful MAC is dominated by PARTIAL SUMS when the array is
    small relative to K (many K-tiles). A bigger array (fewer K-tiles) shrinks it.
"""
import argparse
import json
import os
import re
import subprocess
import sys

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
ROOT = os.path.join(os.path.dirname(__file__), "..")
RTL = ["pe", "skew_delay", "systolic_array", "sa_core", "sram_sdp", "lrb", "perf_counters",
       "rst_sync", "dnn_ppu", "dnn_ctrl", "dnn_accel_core", "dnn_reg_if", "dnn_accel_top"]

# (rows, batch)
DEFAULT = [(4, 8), (4, 32), (8, 8), (8, 32), (16, 32)]


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode:
        print(open(log).read()[-2000:])
        sys.exit(f"command failed: {' '.join(cmd[:4])} ... (see {log})")


def hw_for(rows, batch, k_max=64, n_max=16):
    kt = -(-k_max // rows)
    nt = max(-(-n_max // rows), 1)
    return dict(rows=rows, m_max=batch, kt_max=kt, nt_max=nt, act_depth=2 * batch * kt,
                acc_depth=batch * nt, lrb_depth=batch, wgt_depth=512, bias_depth=16, num_layers=4)


def one(rows, batch, nbatches=3):
    hw = hw_for(rows, batch)
    d = os.path.join(BUILD, "sweep", f"r{rows}_m{batch}")
    os.makedirs(d, exist_ok=True)
    args = ["python3", "sw/compile.py", "--out", d, "--rows", str(rows), "--batch", str(batch),
            "--nbatches", str(nbatches)] + sum([["--" + k.replace("_", "-"), str(v)]
                                                 for k, v in hw.items() if k != "rows"], [])
    run(args, d + "/compile.log")
    g = [f"-GROWS={rows}", f"-GCOLS={rows}", f"-GM_MAX={hw['m_max']}", f"-GKT_MAX={hw['kt_max']}",
         f"-GNT_MAX={hw['nt_max']}", f"-GACT_DEPTH={hw['act_depth']}", f"-GWGT_DEPTH={hw['wgt_depth']}",
         f"-GBIAS_DEPTH={hw['bias_depth']}", f"-GACC_DEPTH={hw['acc_depth']}",
         f"-GLRB_DEPTH={hw['lrb_depth']}", f"-GNUM_LAYERS={hw['num_layers']}"]
    run(["verilator", "--binary", "--timing", "-DSTROBES", "-Wno-fatal", "-Wno-lint", "-Wno-style", "-Wno-WIDTH",
         "--top-module", "tb_dnn_player"] + g + ["--Mdir", d + "/obj", "tb/tb_dnn_player.v"]
        + [f"rtl/{m}.v" for m in RTL], d + "/build.log")
    out = subprocess.run([d + "/obj/Vtb_dnn_player", f"+prog={d}/prog.hex", f"+log={d}/rb.txt"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    if "PLAYER PASSED" not in out:
        sys.exit(f"RTL run failed for rows={rows} batch={batch}:\n{out[-800:]}")
    strobes = re.search(r"STROBES acc_wr_en=(\d+) acc_rd_en=(\d+)", out)
    if not strobes:
        sys.exit("no STROBES line in the testbench output (was it built with -DSTROBES?)")

    meta = json.load(open(d + "/prog_meta.json"))
    vals = [int(l.split()[3], 16) for l in open(d + "/rb.txt")]
    reads = meta["reads"]
    vals = vals[len(vals) - len(reads):]
    perf = {}
    for rd, v in zip(reads, vals):
        if rd["kind"] == "perf" and rd["tag"] == nbatches - 1:
            perf[rd["name"]] = v

    M, R = batch, rows
    useful = sum(L["K"] * L["N"] for L in meta["layers"])            # MACs per image, no padding
    padded = sum(L["Kp"] * L["Np"] for L in meta["layers"])
    acc_words = sum((L["KT"] - 1) * M * L["NT"] for L in meta["layers"])   # per direction
    wb = perf["wgt_rd"] * R               # weight word = COLS x 1 byte
    ab = perf["act_rd"] * R               # activation word = ROWS x 1 byte
    lb = perf["lrb_rd"] * R
    pb = 2 * acc_words * R * 4            # read + write, ROWS x 4 bytes
    macs = M * useful
    expect = acc_words * nbatches
    if (int(strobes.group(1)), int(strobes.group(2))) != (expect, expect):
        sys.exit(f"partial-sum formula MISMATCH for rows={rows} batch={batch}: formula {expect}, "
                 f"RTL strobes wr={strobes.group(1)} rd={strobes.group(2)}")
    return dict(rows=R, batch=M, cyc_img=perf["cycles"] / M, mac_cyc=macs / perf["cycles"],
                util=100.0 * macs / perf["cycles"] / (R * R), useful_frac=100.0 * useful / padded,
                wgt_B=wb / macs, act_B=(ab + lb) / macs, psum_B=pb / macs,
                tot_B=(wb + ab + lb + pb) / macs, kt=meta["layers"][0]["KT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int)
    ap.add_argument("--batch", type=int)
    a = ap.parse_args()
    cfgs = [(a.rows, a.batch)] if a.rows and a.batch else DEFAULT
    rows = [one(r, m) for r, m in cfgs]
    print(f"{'array':>7} {'batch':>5} {'K-tiles':>7} {'cyc/img':>8} {'MAC/cyc':>8} {'util%':>6} "
          f"{'wgt B/MAC':>10} {'act B/MAC':>10} {'psum B/MAC':>11} {'total B/MAC':>12}")
    for x in rows:
        print(f"{x['rows']:>3}x{x['rows']:<3} {x['batch']:>5} {x['kt']:>7} {x['cyc_img']:>8.1f} "
              f"{x['mac_cyc']:>8.2f} {x['util']:>6.1f} {x['wgt_B']:>10.3f} {x['act_B']:>10.3f} "
              f"{x['psum_B']:>11.3f} {x['tot_B']:>12.3f}")
    print("\nMAC = useful multiply-accumulates only (zero padding not counted).")
    print("B/MAC = bytes read from on-chip memories (weight SRAM, activation SRAM+reuse buffer,")
    print("accumulator SRAM read+write) per useful MAC. A design with NO reuse fetches 2 B/MAC.")
    print(f"Partial-sum formula == RTL acc_wr_en/acc_rd_en strobe counts, checked for all {len(rows)} configuration(s).")


if __name__ == "__main__":
    main()
