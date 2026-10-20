#!/usr/bin/env python3
"""Step 1 of the software flow: train a small MLP in PyTorch and export ONNX.

What this teaches
-----------------
* A neural-network "model" is just a list of weight matrices + biases + a few
  simple element-wise functions (ReLU). Training finds the numbers; the
  accelerator only ever runs *inference* with the numbers frozen.
* PyTorch is where the numbers are created. ONNX is the neutral file format
  ("a PDF for neural networks") that hands them to tools that are not PyTorch -
  such as our quantizer and, eventually, the hardware compiler.

Model: 64 inputs (8x8 digit image) -> 16 hidden (ReLU) -> 10 outputs (digits).
Data : scikit-learn's bundled `digits` set (1797 images), so no download needed.

Run:  python3 sw/train_mlp.py            (writes build/model.onnx, build/data.npz)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(__file__), "..", "build")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

HIDDEN = [int(v) for v in os.environ.get("HIDDEN", "16").split(",")]   # hidden widths, e.g. HIDDEN=32,16

d = load_digits()
x = (d.data / 16.0).astype(np.float32)          # pixels 0..16 -> 0..1
y = d.target.astype(np.int64)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
print(f"train {x_tr.shape}, test {x_te.shape}")

dims = [64] + HIDDEN + [10]
mods = []
for i in range(len(dims) - 1):
    mods.append(nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2:
        mods.append(nn.ReLU())                 # ReLU after every layer except the last
model = nn.Sequential(*mods)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
lossf = nn.CrossEntropyLoss()
xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)

for epoch in range(300):                        # tiny problem: full-batch is fine
    opt.zero_grad()
    loss = lossf(model(xt), yt)
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == 299:
        with torch.no_grad():
            acc = (model(torch.from_numpy(x_te)).argmax(1).numpy() == y_te).mean()
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  test acc {acc*100:.2f}%")

model.eval()
torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))

# Export to ONNX. (dynamo=False = classic exporter: plain Gemm + Relu nodes.)
onnx_path = os.path.join(OUT, "model.onnx")
torch.onnx.export(model, torch.zeros(1, 64), onnx_path, input_names=["x"],
                  output_names=["logits"], opset_version=13, dynamo=False,
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}})
np.savez(os.path.join(OUT, "data.npz"), x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
print("wrote", onnx_path)
