#!/usr/bin/env python3
"""Random-network stress program generator (verification, not a demo).

Generates many random multi-layer INT8 networks with random shapes, batch
sizes, biases, requant parameters and ReLU flags, computes the bit-exact
golden output, and emits one register program that tb_dnn_player replays.
Also emits negative tests (illegal descriptors must raise `err`).

  python3 sw/gen_random_net.py --rows 8 --ntests 40 --seed 1 --out build/rand_r8.hex \
          --m-max 8 --kt-max 6 --nt-max 6 --act-depth 128 --acc-depth 48 --lrb-depth 16
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dnn_cfg import *
import compiler_lib as cl
import golden


def rand_net(rng, cfg, M):
    """Random legal network for batch M, or None if it does not fit."""
    R = cfg.rows
    nl = int(rng.integers(1, cfg.num_layers + 1))
    words = tiles_n = 0
    dims_k = int(rng.integers(1, cfg.kt_max * R + 1))
    layers = []
    x = rng.integers(-128, 128, size=(M, dims_k)).astype(np.int8)
    h = x
    for i in range(nl):
        K = h.shape[1]
        N = int(rng.integers(1, cfg.nt_max * R + 1))
        KT, NT = -(-K // R), -(-N // R)
        if (KT > cfg.kt_max or NT > cfg.nt_max or M * NT > cfg.acc_depth
                or M * KT > cfg.bank or M * NT > cfg.bank
                or words + KT * NT * R > cfg.wgt_depth or tiles_n + NT > cfg.bias_depth):
            break
        words += KT * NT * R; tiles_n += NT
        kind = rng.integers(0, 6)
        if kind == 0:   w = rng.integers(-128, 128, size=(K, N))
        elif kind == 1: w = np.full((K, N), -128)                       # worst-case magnitude
        elif kind == 2: w = rng.choice([-128, 127, 0, 1, -1], size=(K, N))
        else:           w = rng.integers(-40, 41, size=(K, N))
        b = rng.integers(-2 ** 18, 2 ** 18, size=N) if rng.random() < 0.8 else np.zeros(N, np.int64)
        relu = bool(rng.integers(0, 2))
        acc = h.astype(np.int64) @ w.astype(np.int64) + b
        if np.abs(acc).max() >= 2 ** 31:
            break
        # choose (mult, shift) so outputs land mid-range most of the time
        mult = int(rng.integers(1, 65536))
        scale = max(float(np.abs(acc).mean()), 1.0) * mult / 60.0
        shift = int(np.clip(round(np.log2(scale)), 0, 47))
        if rng.random() < 0.1:
            shift = int(rng.integers(0, 48))                             # also saturating cases
        L = dict(w_q=w.astype(np.int8), bias_q=b, mult=mult, shift=shift, relu=relu)
        layers.append(L)
        h = golden.layer(h, L["w_q"], L["bias_q"], mult, shift, relu)
    return (x, layers) if layers else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--ntests", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    for k in ("act_depth", "wgt_depth", "bias_depth", "acc_depth", "lrb_depth",
              "num_layers", "m_max", "kt_max", "nt_max"):
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=getattr(HwCfg, k))
    a = ap.parse_args()
    cfg = HwCfg(rows=a.rows, act_depth=a.act_depth, wgt_depth=a.wgt_depth,
                bias_depth=a.bias_depth, acc_depth=a.acc_depth, lrb_depth=a.lrb_depth,
                num_layers=a.num_layers, m_max=a.m_max, kt_max=a.kt_max, nt_max=a.nt_max)
    rng = np.random.default_rng(a.seed)

    ops = list(cl.probe_ops(cfg))
    n_ok = n_bad = n_cnt = 0
    stats = dict(layers=0, m1=0, kt1=0, nt1=0, lrb_partial=0, lrb_full=0, relu=0, norelu=0)
    tries = 0
    while n_ok < a.ntests and tries < a.ntests * 20:
        tries += 1
        M = int(rng.integers(1, cfg.m_max + 1))
        r = rand_net(rng, cfg, M)
        if r is None:
            continue
        x, layers = r
        P = cl.plan(layers, cfg, M)
        exp = cl.expected_outputs(P, x)
        ops += cl.static_ops(P, cfg)
        # STATUS read right after START must be exactly busy | rd_valid (0x9): no stale done/err
        ops += cl.input_ops(P, cfg, x) + [(OP_WRITE, REG_CTRL, 1),
                                          (OP_READ_EXPECT, REG_STATUS, STATUS_BUSY | STATUS_RD_VALID),
                                          (OP_WAIT_DONE, 0, 0)]
        ops += cl.readback_ops(P, cfg, exp[-1], [], n_ok)
        n_ok += 1
        stats["layers"] += len(P)
        stats["m1"] += M == 1
        for L in P:
            stats["kt1"] += L["KT"] == 1; stats["nt1"] += L["NT"] == 1
            stats["relu"] += L["relu"]; stats["norelu"] += not L["relu"]
        stats["lrb_full"] += M <= cfg.lrb_depth
        stats["lrb_partial"] += M > cfg.lrb_depth

        # every 4th test: illegal commands that must be rejected (err, not done)
        if n_ok % 4 == 0:
            # (a) bad layer count. The descriptors are still the legal ones of the run above, so a
            #     wrongly accepted count (e.g. 9 wrapping to 1 in a 3-bit register) would run and NOT
            #     raise err - the test can only pass if the count itself is rejected.
            lay_w = clog2(cfg.num_layers + 1)
            bad_n = [0, cfg.num_layers + 1, (1 << lay_w) + 1, (1 << 16) + 1]   # last two would wrap to 1
            ops += [(OP_WRITE, REG_NLAYERS, bad_n[n_cnt % len(bad_n)]),        # rotate: all four get used
                    (OP_WRITE, REG_CTRL, 1), (OP_WAIT_DONE_ERR, 0, 0),
                    (OP_WRITE, REG_NLAYERS, len(P))]                           # restore a legal count
            n_cnt += 1
            n_bad += 1
            # (b) illegal descriptor field(s) in layer 0
            bad = [[(F_M, 0)], [(F_M, cfg.m_max + 1)], [(F_KT, 0)], [(F_KT, cfg.kt_max + 1)],
                   [(F_NT, 0)], [(F_NT, cfg.nt_max + 1)], [(F_SHIFT, 48)]]
            if cfg.m_max * cfg.nt_max > cfg.acc_depth:      # each field legal, product too big
                bad.append([(F_M, cfg.m_max), (F_NT, cfg.nt_max)])
            fields = bad[int(rng.integers(0, len(bad)))]
            ops += [(OP_WRITE, DESC_BASE + 0 * 16 + f, v) for f, v in fields]
            ops += [(OP_WRITE, REG_CTRL, 1), (OP_WAIT_DONE_ERR, 0, 0)]
            n_bad += 1

    with open(a.out, "w") as f:
        f.write("\n".join(cl.to_hex_lines(ops)) + "\n")
    print(f"rows={a.rows}: {n_ok} random networks + {n_bad} negative tests, {len(ops)} ops; "
          f"coverage {stats}")


if __name__ == "__main__":
    main()
