#!/usr/bin/env python3
"""Step 5: turn the RTL's own readback into an accuracy / performance report.

Reads build/rtl_readback.txt (written by tb_dnn_player) and build/prog_meta.json.
The accuracy printed here is computed ONLY from values the RTL returned - it is
the accuracy of the hardware, not of the Python model.

Also prints the throughput / data-reuse numbers (measured by the RTL's own
performance counters) and a roofline-style summary.
"""
import json
import os
import sys
import numpy as np

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")


def s8(v):
    return v - 256 if v >= 128 else v


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "rtl_readback.txt")
    meta = json.load(open(os.path.join(BUILD, "prog_meta.json")))
    q = np.load(os.path.join(BUILD, "qnet.npz"))
    rows = [l.split() for l in open(log_path)]
    vals = [int(r[3], 16) for r in rows]
    reads = meta["reads"]
    vals = vals[len(vals) - len(reads):]            # skip the static ID/CFG reads
    R, nb, B = meta["rows"], meta["nbatches"], meta["batch"]
    n_pad = meta["last_layer_np"]

    logits = np.zeros((nb * B, n_pad), np.int8)
    perf = {k: [] for k in ("cycles", "act_rd", "lrb_rd", "wgt_rd", "vec")}
    for rd, v in zip(reads, vals):
        if rd["kind"] == "out":
            m = rd["tag"] * B + rd["m"]
            for c in range(R):                       # 4 INT8 lanes per 32-bit word (rows==4)
                logits[m, rd["nt"] * R + c] = s8((v >> (8 * c)) & 0xFF)
        else:
            perf[rd["name"]].append(v)

    labels = np.array(meta["labels"])
    n = meta["last_layer_n"]
    rtl_acc = (logits[:, :n].argmax(1) == labels).mean()
    print("=" * 66)
    print(f"images classified by the RTL     : {nb * B}")
    print(f"float model accuracy (PyTorch)   : {float(q['f_acc'])*100:6.2f} %")
    print(f"INT8 python golden accuracy      : {float(q['q_acc'])*100:6.2f} %")
    print(f"INT8 accuracy measured FROM RTL  : {rtl_acc*100:6.2f} %")
    print("=" * 66)

    cyc = np.array(perf["cycles"])
    pe = R * R
    useful = B * sum(L["K"] * L["N"] for L in meta["layers"])      # real MACs, no zero padding
    padded = meta["macs_per_batch"]                                 # incl. zero-padded rows/cols
    print(f"per batch of {B} images (cycles from the RTL perf counters):")
    print(f"  busy cycles          : {cyc.mean():8.0f}   ({cyc.mean()/B:.0f} cycles/image)")
    print(f"  useful MACs          : {useful:8d}   ({useful/cyc.mean():.2f} MAC/cycle, peak {pe})")
    print(f"  (incl. zero padding) : {padded:8d}   ({padded/cyc.mean():.2f} MAC/cycle)")
    print(f"  array utilization    : {100*useful/cyc.mean()/pe:8.1f} %   (useful MACs / (cycles x {pe} PEs))")
    print(f"  weight-SRAM reads    : {np.mean(perf['wgt_rd']):8.0f} words")
    print(f"  act-SRAM reads       : {np.mean(perf['act_rd']):8.0f} words")
    print(f"  LRB reads (reuse)    : {np.mean(perf['lrb_rd']):8.0f} words")
    tot = np.mean(perf["act_rd"]) + np.mean(perf["lrb_rd"])
    print(f"  activation fetches served by the LRB : {100*np.mean(perf['lrb_rd'])/tot:5.1f} %"
          f"  (act-SRAM reads cut {tot/np.mean(perf['act_rd']):.2f}x)")
    # data movement per useful MAC (partial-sum traffic derived from tiling geometry)
    acc_words = sum((L["KT"] - 1) * B * L["NT"] for L in meta["layers"])   # writes == reads
    w_b = np.mean(perf["wgt_rd"]) * R
    a_b = tot * R
    p_b = 2 * acc_words * R * 4
    print(f"  bytes moved per useful MAC: weights {w_b/useful:.3f} + activations {a_b/useful:.3f} "
          f"+ partial sums {p_b/useful:.3f} = {(w_b+a_b+p_b)/useful:.3f}   (no-reuse design: 2.000)")
    print(f"  at 100 MHz: {B/(cyc.mean()*10e-9):9.0f} images/s   (pre-synthesis estimate)")
    print("=" * 66)
    sys.exit(0 if abs(rtl_acc - float(q["q_acc"])) < 1e-9 else 1)


if __name__ == "__main__":
    main()
