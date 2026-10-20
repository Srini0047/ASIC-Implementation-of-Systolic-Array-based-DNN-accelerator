#!/usr/bin/env python3
"""Follow ONE image through the quantized network, printing every number.

  python3 sw/trace_one.py            # test image 0, neuron 0 of layer 1
  python3 sw/trace_one.py 17 5       # test image 17, neuron 5

This is a learning tool: it shows, with real values from build/qnet.npz, exactly
what the hardware does - which multiplications feed which sum, where the bias
enters, what (mult, shift) do, where ReLU acts and what shape every tensor has.
The final numbers are the ones the RTL produces (they are bit-exact).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import golden

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")


def main():
    img = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    neuron = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    z = np.load(os.path.join(BUILD, "qnet.npz"))
    d = np.load(os.path.join(BUILD, "data.npz"))
    L = [dict(w_q=z[f"L{i}_w_q"], bias_q=z[f"L{i}_bias_q"], mult=int(z[f"L{i}_mult"]),
              shift=int(z[f"L{i}_shift"]), relu=bool(z[f"L{i}_relu"]),
              s_w=float(z[f"L{i}_s_w"]), s_in=float(z[f"L{i}_s_in"]), s_out=float(z[f"L{i}_s_out"]))
         for i in range(int(z["n_layers"]))]
    x_f = d["x_test"][img]                       # float 0..1  (pixel/16)
    x_q = z["x_q_test"][img]                     # int8
    label = int(d["y_test"][img])

    print(f"=== test image #{img}  (true digit: {label}) ===\n")
    print("Step 0 - the input. An 8x8 grayscale image, pixel values 0..16 (this is the 'tensor'):")
    for r in range(8):
        print("   " + " ".join(f"{int(round(v * 16)):2d}" for v in x_f[r * 8:(r + 1) * 8]))
    print(f"\n   flattened -> vector of {x_q.size} numbers; shape [1, 64]  (batch of 1)")
    print(f"   quantized:  x_q = round(x / s_in),  s_in = {L[0]['s_in']:.5f}  ->  int8 in [-128,127]")
    print(f"   x_q[0:12] = {x_q[:12].tolist()} ...\n")

    w, b = L[0]["w_q"], L[0]["bias_q"]
    K, N = w.shape
    print(f"Step 1 - layer 1: {K} inputs -> {N} neurons.  Weight tensor W1 shape [{K}, {N}] (int8), bias [{N}] (int32).")
    print(f"   Neuron {neuron} = one COLUMN of W1. It needs {K} multiplications and {K-1} additions:")
    prods = x_q.astype(np.int64) * w[:, neuron].astype(np.int64)
    for k in range(5):
        print(f"      k={k:2d}:  x_q[{k}] * W1[{k},{neuron}] = {int(x_q[k]):4d} * {int(w[k, neuron]):4d} = {int(prods[k]):6d}")
    print(f"      ...    ({K - 5} more products)")
    acc = int(prods.sum())
    print(f"   sum of all {K} products (INT32 accumulator)        = {acc}")
    v = acc + int(b[neuron])
    print(f"   + bias[{neuron}]                                  = {acc} + {int(b[neuron])} = {v}")
    m, sh = L[0]["mult"], L[0]["shift"]
    prod = v * m
    print(f"   x mult ({m})                                = {prod}")
    print(f"   + rounding (1 << {sh - 1})  then  >> {sh}              = {(prod + (1 << (sh - 1))) >> sh}")
    y = int(golden.requant(np.array([v]), m, sh, L[0]["relu"])[0])
    print(f"   ReLU/clamp to [0,127] (layer 1 has ReLU)          = {y}   <- this is the neuron's INT8 output")
    print(f"   (real-number meaning: {y} x s_out({L[0]['s_out']:.5f}) = {y * L[0]['s_out']:.4f})\n")

    outs = golden.run_net(L, x_q[None, :])
    h1 = outs[0][0]
    print(f"   all {N} neurons of layer 1 (int8): {h1.tolist()}   shape [1, {N}]")
    print(f"   -> {int((h1 == 0).sum())} of {N} are exactly 0: that is what ReLU does (negative -> 0)\n")

    for i in range(1, len(L)):
        wi = L[i]["w_q"]
        last = i == len(L) - 1
        n_real = 10 if last else wi.shape[1]
        out = outs[i][0][:n_real]
        kind = "scores" if last else "outputs"
        print(f"Step {i + 1} - layer {i + 1}: {wi.shape[0]} -> {n_real}. W{i + 1} shape [{wi.shape[0]}, {wi.shape[1]}], "
              f"{'ReLU' if L[i]['relu'] else 'no ReLU'}.")
        print(f"   {kind} (int8): {out.tolist()}   shape [1, {n_real}]")
    y2 = outs[-1][0][:10]
    print(f"   argmax = {int(y2.argmax())}   (true digit {label})  {'CORRECT' if int(y2.argmax()) == label else 'wrong'}\n")

    R = 4
    meta_path = os.path.join(BUILD, "prog_meta.json")
    if os.path.exists(meta_path):
        import json
        R = json.load(open(meta_path))["rows"]
    print(f"Hardware view of the same computation ({R}x{R} array):")
    total = 0
    for i, Li in enumerate(L):
        k, n = Li["w_q"].shape
        n_real = 10 if i == len(L) - 1 else n
        kt, nt = -(-k // R), -(-n_real // R)
        total += k * n_real
        print(f"   layer {i + 1}: {k}x{n_real} weights = {kt} K-tiles x {nt} N-tiles = {kt * nt} passes of the array"
              f"   ({k * n_real} MACs)")
    print(f"   each pass = load one {R}x{R} weight tile, then stream the batch through it.")
    print(f"   MACs for this image: {total}")


if __name__ == "__main__":
    main()
