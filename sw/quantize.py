#!/usr/bin/env python3
"""Step 2: read the ONNX model and turn floats into the integers the hardware uses.

The idea, in hardware terms
---------------------------
A float number like 0.37 needs a 32-bit floating-point multiplier (big, slow,
power-hungry). If instead we agree that "1 LSB of this 8-bit number means
0.0031", we can store 0.37 as the integer 119 and do all the maths with cheap
integer multipliers. That agreement (one scale factor per tensor) is
*quantization*:

        real_value  ~=  scale * int_value

Symmetric per-tensor INT8:
        s_w   = max|W| / 127            w_q = round(W / s_w)      in [-127,127]
        s_x   = max(x)  / 127           x_q = round(x / s_x)      (input)
        s_out = max(y)  / 127           (from a calibration run of the float net)
        bias_q = round(b / (s_x * s_w))                           INT32

After the integer matmul the accumulator is in units of (s_x*s_w). To feed the
next layer we must re-express it in units of s_out:
        y_q = acc * (s_x*s_w / s_out)          <- a real number, e.g. 0.0043
The hardware cannot multiply by 0.0043, so we approximate it as mult / 2^shift
with a 16-bit integer `mult` - that is the (mult, shift) pair the PPU uses.

Run: python3 sw/quantize.py   (needs build/model.onnx and build/data.npz)
Writes build/qnet.npz and prints float vs INT8 accuracy.
"""
import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
import golden

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")


def load_onnx_layers(path):
    """Return [(W[out,in], b[out], relu_after)] from a Gemm(+Relu) ONNX graph."""
    m = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    nodes = list(m.graph.node)
    layers = []
    for i, n in enumerate(nodes):
        if n.op_type == "Gemm":
            w, b = init[n.input[1]], init[n.input[2]]
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            if not trans_b:
                w = w.T                          # make it [out,in]
            relu = i + 1 < len(nodes) and nodes[i + 1].op_type == "Relu"
            layers.append((w.astype(np.float64), b.astype(np.float64), relu))
        elif n.op_type not in ("Relu", "Flatten", "Identity"):
            raise SystemExit(f"unsupported ONNX op {n.op_type} (this flow: Gemm+Relu only)")
    return layers


def choose_mult_shift(m_real, mult_bits=16, shift_max=47):
    """Approximate m_real ~= mult / 2**shift with mult < 2**mult_bits."""
    assert m_real > 0
    shift = int(np.floor(np.log2(((1 << mult_bits) - 1) / m_real)))
    shift = max(0, min(shift_max, shift))
    mult = int(round(m_real * (1 << shift)))
    if mult >= (1 << mult_bits):
        mult = (1 << mult_bits) - 1
    return mult, shift


def main():
    data = np.load(os.path.join(BUILD, "data.npz"))
    x_cal, x_te, y_te = data["x_train"], data["x_test"], data["y_test"]
    fl = load_onnx_layers(os.path.join(BUILD, "model.onnx"))

    # ---- float reference through onnxruntime (proves the ONNX file is the model)
    sess = ort.InferenceSession(os.path.join(BUILD, "model.onnx"))
    f_logits = sess.run(None, {"x": x_te.astype(np.float32)})[0]
    f_acc = (f_logits.argmax(1) == y_te).mean()

    # ---- calibration: run the float net on the training set, record ranges
    h = x_cal.astype(np.float64)
    ranges = []
    for (w, b, relu) in fl:
        h = h @ w.T + b
        if relu:
            h = np.maximum(h, 0)
        ranges.append(np.abs(h).max())

    s_in = float(np.abs(x_cal).max()) / 127.0          # input scale
    layers, s_prev = [], s_in
    for (w, b, relu), rng in zip(fl, ranges):
        s_w = float(np.abs(w).max()) / 127.0
        s_out = float(rng) / 127.0
        w_q = np.clip(np.round(w.T / s_w), -127, 127).astype(np.int8)     # [in,out]
        bias_q = np.round(b / (s_prev * s_w)).astype(np.int64)
        assert np.abs(bias_q).max() < 2 ** 31
        mult, shift = choose_mult_shift(s_prev * s_w / s_out)
        layers.append(dict(w_q=w_q, bias_q=bias_q.astype(np.int32), mult=mult,
                           shift=shift, relu=bool(relu),
                           s_w=s_w, s_in=s_prev, s_out=s_out))
        print(f"layer {len(layers)}: W{w_q.shape} relu={relu} s_w={s_w:.5f} "
              f"s_out={s_out:.5f} mult={mult} shift={shift} "
              f"(real {s_prev*s_w/s_out:.6f} ~ {mult/2**shift:.6f})")
        s_prev = s_out

    # ---- integer-only inference on the test set
    x_q = np.clip(np.round(x_te / s_in), -128, 127).astype(np.int8)
    q_logits = golden.run_net(layers, x_q)[-1]
    q_acc = (q_logits[:, :10].argmax(1) == y_te).mean()
    print(f"float (onnxruntime) accuracy : {f_acc*100:.2f}%")
    print(f"INT8 integer-only accuracy   : {q_acc*100:.2f}%   (this is what the RTL must reproduce)")

    np.savez(os.path.join(BUILD, "qnet.npz"), s_in=s_in, n_layers=len(layers),
             **{f"L{i}_{k}": np.asarray(v) for i, L in enumerate(layers)
                for k, v in L.items()},
             x_q_test=x_q, y_test=y_te, f_acc=f_acc, q_acc=q_acc)


if __name__ == "__main__":
    main()
