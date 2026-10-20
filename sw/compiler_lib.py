"""The accelerator "compiler": quantized layers -> memory images + register program.

This is the software half of the hardware/software contract. It decides
    * how a big weight matrix is cut into ROWS x COLS tiles (padding with zeros),
    * where every tile / bias / activation lives in the on-chip SRAMs,
    * what each layer descriptor contains,
    * the exact sequence of register writes a CPU must perform.
The C driver in driver/ implements the SAME sequence; scripts/check_driver.sh
proves the two produce identical register streams.
"""
import numpy as np
from dnn_cfg import *
import golden


def ceil_div(a, b):
    return -(-a // b)


def pad_layers(layers, cfg):
    """Zero-pad K and N of every layer up to a multiple of cfg.rows."""
    R, out = cfg.rows, []
    for L in layers:
        K, N = L["w_q"].shape
        Kp, Np = ceil_div(K, R) * R, ceil_div(N, R) * R
        w = np.zeros((Kp, Np), np.int8); w[:K, :N] = L["w_q"]
        b = np.zeros(Np, np.int64); b[:N] = L["bias_q"]
        out.append(dict(w_q=w, bias_q=b, mult=int(L["mult"]), shift=int(L["shift"]),
                        relu=bool(L["relu"]), K=K, N=N, Kp=Kp, Np=Np))
    return out


def plan(layers, cfg, batch):
    """Compute descriptors and memory bases; validate against the hardware."""
    P = pad_layers(layers, cfg)
    if len(P) > cfg.num_layers:
        raise ValueError(f"{len(P)} layers > hardware NUM_LAYERS={cfg.num_layers}")
    if not 1 <= batch <= cfg.m_max:
        raise ValueError(f"batch {batch} not in 1..M_MAX={cfg.m_max}")
    wptr = bptr = 0
    for i, L in enumerate(P):
        L["KT"], L["NT"] = L["Kp"] // cfg.rows, L["Np"] // cfg.cols
        L["M"] = batch
        L["src"] = 0 if i % 2 == 0 else cfg.bank
        L["dst"] = cfg.bank if i % 2 == 0 else 0
        L["wbase"], L["bbase"] = wptr, bptr
        wptr += L["KT"] * L["NT"] * cfg.rows
        bptr += L["NT"]
        chk = [(L["KT"] <= cfg.kt_max, "KT > KT_MAX"), (L["NT"] <= cfg.nt_max, "NT > NT_MAX"),
               (batch * L["KT"] <= cfg.bank, "input does not fit one activation region"),
               (batch * L["NT"] <= cfg.bank, "output does not fit one activation region"),
               (batch * L["NT"] <= cfg.acc_depth, "M*NT > ACC_DEPTH (accumulator SRAM)"),
               (L["shift"] <= 47 and L["mult"] < 65536, "mult/shift out of range")]
        for ok, msg in chk:
            if not ok:
                raise ValueError(f"layer {i}: {msg}")
        if i > 0 and P[i - 1]["Np"] != L["Kp"]:
            raise ValueError(f"layer {i}: K does not match previous N")
    if wptr > cfg.wgt_depth:
        raise ValueError(f"weights need {wptr} words > WGT_DEPTH={cfg.wgt_depth}")
    if bptr > cfg.bias_depth:
        raise ValueError(f"bias needs {bptr} words > BIAS_DEPTH={cfg.bias_depth}")
    if 2 * cfg.bank > cfg.act_depth:
        raise ValueError("ACT_DEPTH must hold two regions of M_MAX*KT_MAX words")
    return P


# ---- register-write generators ----------------------------------------------
def win_writes(base, stride, word_idx, word, nbeats):
    return [(OP_WRITE, base + word_idx * stride + b, v)
            for b, v in enumerate(split_beats(word, nbeats))]


def probe_ops(cfg):
    """Discovery reads a driver performs first (ID + hardware configuration)."""
    return [(OP_READ_EXPECT, REG_ID, ID_VALUE),
            (OP_READ_EXPECT, REG_CFG_ARRAY,
             (cfg.rows << 24) | (cfg.m_max << 16) | (cfg.kt_max << 8) | cfg.nt_max),
            (OP_READ_EXPECT, REG_CFG_LAYERS, cfg.num_layers),
            (OP_READ_EXPECT, REG_CFG_ACT, cfg.act_depth),
            (OP_READ_EXPECT, REG_CFG_WGT, cfg.wgt_depth),
            (OP_READ_EXPECT, REG_CFG_BIAS, cfg.bias_depth),
            (OP_READ_EXPECT, REG_CFG_ACC, cfg.acc_depth),
            (OP_READ_EXPECT, REG_CFG_LRB, cfg.lrb_depth)]


def static_ops(P, cfg):
    """Everything that does not change between input batches."""
    ops = []
    R = cfg.rows
    for i, L in enumerate(P):
        # weight tiles in pass order p = kt*NT + nt, ROWS words per tile
        w = L["w_q"]
        p = 0
        for kt in range(L["KT"]):
            for nt in range(L["NT"]):
                for r in range(R):
                    lanes = w[kt * R + r, nt * cfg.cols:(nt + 1) * cfg.cols]
                    ops += win_writes(WGT_BASE, cfg.stride_w, L["wbase"] + p * R + r,
                                      pack_lanes(lanes, 8), cfg.beats_w)
                p += 1
        for nt in range(L["NT"]):
            lanes = L["bias_q"][nt * cfg.cols:(nt + 1) * cfg.cols]
            ops += win_writes(BIAS_BASE, cfg.stride_b, L["bbase"] + nt,
                              pack_lanes(lanes, 32), cfg.beats_b)
        fields = {F_M: L["M"], F_KT: L["KT"], F_NT: L["NT"], F_WBASE: L["wbase"],
                  F_BBASE: L["bbase"], F_SRC: L["src"], F_DST: L["dst"],
                  F_MULT: L["mult"], F_SHIFT: L["shift"], F_RELU: int(L["relu"])}
        for f, v in fields.items():
            ops.append((OP_WRITE, DESC_BASE + i * 16 + f, v))
    ops.append((OP_WRITE, REG_NLAYERS, len(P)))
    return ops


def input_ops(P, cfg, x_q):
    """x_q [M, K] int8 -> writes into the activation window (layer-0 region)."""
    L0, R = P[0], cfg.rows
    M = x_q.shape[0]
    xp = np.zeros((M, L0["Kp"]), np.int8); xp[:, :x_q.shape[1]] = x_q
    ops = []
    for m in range(M):
        for kt in range(L0["KT"]):
            word = pack_lanes(xp[m, kt * R:(kt + 1) * R], 8)
            ops += win_writes(ACT_BASE, cfg.stride_a, L0["src"] + m * L0["KT"] + kt,
                              word, cfg.beats_a)
    return ops


def run_ops():
    return [(OP_WRITE, REG_CTRL, 1), (OP_WAIT_DONE, 0, 0)]


def readback_ops(P, cfg, expected_last, reads, tag):
    """Read the last layer's output words; expected_last [M, Np] int8."""
    L = P[-1]
    R, M = cfg.rows, expected_last.shape[0]
    ops = []
    for m in range(M):
        for nt in range(L["NT"]):
            lanes = expected_last[m, nt * R:(nt + 1) * R]
            word = pack_lanes(lanes, 8)
            ops.append((OP_WRITE, REG_ACT_RD_ADDR, L["dst"] + m * L["NT"] + nt))
            ops.append((OP_WAIT_RD_VALID, 0, 0))
            for b, v in enumerate(split_beats(word, cfg.beats_a)):
                ops.append((OP_READ_EXPECT, REG_ACT_RD_DATA + b, v))
                reads.append(dict(kind="out", tag=tag, m=m, nt=nt, beat=b))
    return ops


def perf_ops(reads, tag):
    ops = []
    for name, reg in (("cycles", REG_PERF_CYCLES), ("act_rd", REG_PERF_ACT_RD),
                      ("lrb_rd", REG_PERF_LRB_RD), ("wgt_rd", REG_PERF_WGT_RD),
                      ("vec", REG_PERF_VEC)):
        ops.append((OP_READ_LOG, reg, 0))
        reads.append(dict(kind="perf", tag=tag, name=name))
    return ops


def to_hex_lines(ops):
    return [f"{op:02x}00{addr:04x}{data & 0xFFFFFFFF:08x}" for op, addr, data in ops + [(OP_END, 0, 0)]]


def expected_outputs(P, x_q):
    """Golden per-layer INT8 outputs on the PADDED network (bit-exact to RTL)."""
    M = x_q.shape[0]
    xp = np.zeros((M, P[0]["Kp"]), np.int8); xp[:, :x_q.shape[1]] = x_q
    return golden.run_net(P, xp)
