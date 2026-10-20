#!/usr/bin/env python3
"""Smallest useful experiment: a 4->4 layer you can verify with a pencil.

  python3 sw/example_tiny_net.py        # writes build/tiny_prog.hex, prints the expected outputs
  bash scripts/run_tiny.sh              # runs it on the RTL (PASS = RTL equals the numbers below)

One layer, one 4x4 weight tile, batch of 2 inputs, no quantization tricks
(mult = 1, shift = 0 means "output = accumulator", so you can check by hand):

        W = [[ 1  2  3  4]          x0 = [1 2 3 4]        bias = [0 0 0 -200]
             [ 5  6  7  8]          x1 = [1 0 0 0]        ReLU on
             [ 9 10 11 12]
             [13 14 15 16]]

  y0 = 1*[1 2 3 4] + 2*[5 6 7 8] + 3*[9 10 11 12] + 4*[13 14 15 16] = [90 100 110 120] + bias -> [90 100 110 -80] -> ReLU [90 100 110 0]
  y1 = 1*[1 2 3 4]                                                   = [ 1   2   3   4] + bias -> [ 1   2   3 -196] -> ReLU [ 1 2 3 0]

Edit W, x, bias, mult, shift, relu and re-run to build your own experiments.
Outputs are int8, so results above 127 saturate to 127 - try it.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dnn_cfg import HwCfg
import compiler_lib as cl

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")

W = np.arange(1, 17, dtype=np.int8).reshape(4, 4)
BIAS = np.array([0, 0, 0, -200], dtype=np.int64)
X = np.array([[1, 2, 3, 4], [1, 0, 0, 0]], dtype=np.int8)
MULT, SHIFT, RELU = 1, 0, True

cfg = HwCfg()                                             # default hardware: 4x4
layers = [dict(w_q=W, bias_q=BIAS, mult=MULT, shift=SHIFT, relu=RELU)]
P = cl.plan(layers, cfg, batch=X.shape[0])
expected = cl.expected_outputs(P, X)[-1]                  # golden, [batch, N]

ops = cl.probe_ops(cfg) + cl.static_ops(P, cfg) + cl.input_ops(P, cfg, X) + cl.run_ops()
ops += cl.readback_ops(P, cfg, expected, [], 0)           # READ_EXPECT: RTL must return these

os.makedirs(BUILD, exist_ok=True)
with open(os.path.join(BUILD, "tiny_prog.hex"), "w") as f:
    f.write("\n".join(cl.to_hex_lines(ops)) + "\n")

print("x  =\n", X)
print("W  =\n", W)
print("bias =", BIAS.tolist(), " mult =", MULT, " shift =", SHIFT, " relu =", RELU)
print("expected outputs (golden model):\n", expected)
print(f"wrote build/tiny_prog.hex ({len(ops)} register operations)")
