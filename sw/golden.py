"""Integer-only reference model - the "golden" the RTL must match BIT-EXACTLY.

This is the same arithmetic as rtl/dnn_ppu.v and the systolic array:

    acc  = sum_k  x_q[k] * w_q[k, n]                 (INT8 x INT8 -> INT32)
    v    = acc + bias_q[n]
    y_q  = clamp( (v * mult + (1 << (shift-1))) >> shift ,  lo, 127 )
           lo = 0 if relu else -128

No floating point anywhere, so any mismatch with the RTL is a real bug.
Python ints / int64 are used so nothing can silently wrap.
"""
import numpy as np


def requant(v, mult, shift, relu):
    """v: int64 array (acc + bias). Mirrors dnn_ppu.v stage 2-3."""
    prod = v.astype(np.int64) * int(mult)
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    q = (prod + rnd) >> shift                   # numpy >> on int64 is arithmetic
    lo = 0 if relu else -128
    return np.clip(q, lo, 127).astype(np.int8)


def layer(x_q, w_q, bias_q, mult, shift, relu):
    """x_q [M,K] int8, w_q [K,N] int8, bias_q [N] int32 -> [M,N] int8."""
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64)
    v = acc + bias_q.astype(np.int64)
    if np.abs(v).max() >= 2 ** 31:
        raise OverflowError("accumulator would overflow INT32 - re-quantize")
    return requant(v, mult, shift, relu)


def run_net(layers, x_q):
    """layers: list of dicts with w_q,bias_q,mult,shift,relu. Returns every
    layer's INT8 output (list), so tests can compare intermediate results."""
    outs = []
    h = x_q
    for L in layers:
        h = layer(h, L["w_q"], L["bias_q"], L["mult"], L["shift"], L["relu"])
        outs.append(h)
    return outs
