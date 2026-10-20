"""Compare the CHIP's answer with the PYTHON answer, for one picture or for all of them.

  python sw\\compare_one.py            picture 0, side by side
  python sw\\compare_one.py 17         picture 17 (0..359)
  python sw\\compare_one.py all        summary over all 360 pictures

Needs only numpy.  Run it AFTER the simulator has written build/rtl_readback.txt.
  chip   = the 10 scores the RTL returned (read back from build/rtl_readback.txt)
  python = the same 10 scores computed here by plain integer arithmetic (sw/golden.py)
The two lists must be identical, number for number.  The biggest score is the guessed digit.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden

BUILD = os.path.join(HERE, "..", "build")
s8 = lambda v: v - 256 if v >= 128 else v

q = np.load(os.path.join(BUILD, "qnet.npz"))
meta = json.load(open(os.path.join(BUILD, "prog_meta.json")))
layers = [dict(w_q=q[f"L{i}_w_q"], bias_q=q[f"L{i}_bias_q"], mult=int(q[f"L{i}_mult"]),
               shift=int(q[f"L{i}_shift"]), relu=bool(q[f"L{i}_relu"])) for i in range(int(q["n_layers"]))]

# ---- Python's answer: run the integer network on all 360 INT8 pictures ----
x_q = q["x_q_test"]
py = golden.run_net(layers, x_q)[-1][:, :10].astype(int)            # [360,10]
labels = np.array(meta["labels"])

# ---- the chip's answer: decode the read-back log (same decoding as sw/report.py) ----
log = os.path.join(BUILD, "rtl_readback.txt")
if not os.path.exists(log):
    sys.exit("build/rtl_readback.txt not found - run the simulator first (Job 1).")
vals = [int(l.split()[3], 16) for l in open(log)]
reads = meta["reads"]
vals = vals[len(vals) - len(reads):]
B, R = meta["batch"], meta["rows"]
chip = np.zeros((meta["nbatches"] * B, meta["last_layer_np"]), int)
for rd, v in zip(reads, vals):
    if rd["kind"] == "out":
        m = rd["tag"] * B + rd["m"]
        for c in range(R):
            chip[m, rd["nt"] * R + c] = s8((v >> (8 * c)) & 0xFF)
chip = chip[:, :10]

same = (chip == py).all(axis=1)
arg = lambda a: int(np.argmax(a))

what = sys.argv[1] if len(sys.argv) > 1 else "0"
if what == "all":
    print("pictures                              :", len(py))
    print("chip == python on all 10 scores       : %d of %d" % (same.sum(), len(py)))
    print("chip guessed the right digit          : %d of %d  (%.2f %%)" %
          ((chip.argmax(1) == labels).sum(), len(py), 100 * (chip.argmax(1) == labels).mean()))
    print("python guessed the right digit        : %d of %d  (%.2f %%)" %
          ((py.argmax(1) == labels).sum(), len(py), 100 * (py.argmax(1) == labels).mean()))
    bad = np.where(~same)[0]
    print("pictures where chip and python differ :", [int(b) for b in bad] if len(bad) else "none")
else:
    i = int(what)
    print("picture %d   (batch %d, position %d in the batch)   correct digit (label) = %d\n" % (i, i // B, i % B, labels[i]))
    print("digit      :  " + " ".join("%5d" % d for d in range(10)))
    print("chip  score:  " + " ".join("%5d" % v for v in chip[i]))
    print("python score: " + " ".join("%5d" % v for v in py[i]))
    print("same?      :  " + " ".join("%5s" % ("ok" if a == b else "DIFF") for a, b in zip(chip[i], py[i])))
    print("\nchip   picks digit %d   |   python picks digit %d   |   label is %d" % (arg(chip[i]), arg(py[i]), labels[i]))
    print("RESULT:", "IDENTICAL - the chip matches Python exactly" if same[i] else "DIFFERENT - there is a mismatch")
