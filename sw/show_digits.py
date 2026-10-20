"""Show the pictures the project uses.   Run from the project folder:   python sw\\show_digits.py

Needs only numpy.  Writes build/digits_preview.html and tries to open it in your browser.
  - Part 1: the 360 TEST pictures, 8 per row = one batch per row (exactly what the chip is fed, batch 0..44).
  - Part 2: 40 TRAINING pictures (the ones the network learned from).
  - Part 3: picture 0 as numbers: the original 0..1 values and the INT8 values that are written into the chip.
"""
import os, sys, webbrowser
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(root, "build", "data.npz"))
q = np.load(os.path.join(root, "build", "qnet.npz"))
xt, yt = d["x_test"], d["y_test"]
xtr, ytr = d["x_train"], d["y_train"]
xq = q["x_q_test"]

def svg(img, size=64):
    r = []
    for i, v in enumerate(img.reshape(-1)):
        g = int(round(255 * (1.0 - float(v))))          # 0.0 = white paper, 1.0 = black ink
        r.append('<rect x="%d" y="%d" width="1" height="1" fill="rgb(%d,%d,%d)"/>' % (i % 8, i // 8, g, g, g))
    return '<svg viewBox="0 0 8 8" width="%d" height="%d" shape-rendering="crispEdges">%s</svg>' % (size, size, "".join(r))

def card(img, label, tag):
    return '<div class="c">%s<div>%s <b>%d</b></div></div>' % (svg(img), tag, label)

h = ['<!doctype html><meta charset="utf-8"><title>Digits used in the project</title>'
     '<style>body{font:15px system-ui,sans-serif;margin:24px;color:#14202a}'
     '.row{display:flex;gap:10px;margin:6px 0;align-items:center;flex-wrap:wrap}'
     '.c{text-align:center;font-size:12px}.c svg{border:1px solid #888;display:block}'
     '.bt{width:70px;font-weight:600}table{border-collapse:collapse}td{border:1px solid #bbb;padding:2px 5px;'
     'font:12px monospace;text-align:right}</style>']
h.append('<h1>The pictures in this project</h1><p>Each picture is 8 x 8 = 64 pixels. Black = ink (value 1.0), '
         'white = paper (0.0). The number under each is the correct answer (the label).</p>')
h.append('<h2>1. The 360 test pictures, one batch of 8 per row</h2><p>Row "batch k" is what the testbench feeds to the chip '
         'in batch k. The chip has never seen these during training.</p>')
for b in range(len(xt) // 8):
    h.append('<div class="row"><span class="bt">batch %d</span>%s</div>' % (
        b, "".join(card(xt[b * 8 + j], yt[b * 8 + j], "#%d:" % (b * 8 + j)) for j in range(8))))
h.append('<h2>2. Forty training pictures</h2><div class="row">%s</div>' % "".join(
    card(xtr[i], ytr[i], "") for i in range(40)))
h.append('<h2>3. Test picture 0 as numbers</h2><p>Left: the original values (0.0 to 1.0). Right: the INT8 values that '
         'the Python flow writes into the chip\'s activation memory (original / %.6f, rounded). Label = <b>%d</b>.</p>' % (float(q["s_in"]), yt[0]))
for title, arr, fmt in (("original", xt[0], "%.3f"), ("INT8 in the chip", xq[0], "%d")):
    h.append('<p>%s</p><table>%s</table>' % (title, "".join(
        "<tr>%s</tr>" % "".join("<td>%s</td>" % (fmt % arr[r * 8 + c]) for c in range(8)) for r in range(8))))
out = os.path.join(root, "build", "digits_preview.html")
open(out, "w", encoding="utf-8").write("".join(h))
print("Wrote", out)
try:
    webbrowser.open("file:///" + out.replace("\\", "/"))
except Exception:
    pass
