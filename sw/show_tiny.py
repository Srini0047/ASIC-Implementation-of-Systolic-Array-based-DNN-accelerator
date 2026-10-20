#!/usr/bin/env python3
"""Decode the tiny example's answer from the simulator's read-back log.

  python sw/example_tiny_net.py       (makes build/tiny_prog.hex and prints the answer you should get)
  ... run the simulator on build/tiny_prog.hex, writing build/custom_readback.txt (or tiny_readback.txt) ...
  python sw/show_tiny.py [logfile]

The last two lines of the log are the two output words; each word holds 4 signed 8-bit numbers.
"""
import os
import sys

BUILD = os.path.join(os.path.dirname(__file__), "..", "build")
paths = [sys.argv[1]] if len(sys.argv) > 1 else [os.path.join(BUILD, "custom_readback.txt"),
                                                os.path.join(BUILD, "tiny_readback.txt")]
path = next((p for p in paths if os.path.exists(p)), None)
if path is None:
    sys.exit("no read-back log found - run the simulator on build/tiny_prog.hex first")
rows = [l.split() for l in open(path)]
for img, r in enumerate(rows[-2:]):
    v = int(r[3], 16)
    lanes = [(v >> (8 * j)) & 0xFF for j in range(4)]
    print(f"image {img}: chip returned word 0x{v:08x} -> numbers {[x - 256 if x >= 128 else x for x in lanes]}")
print("(the unmodified example must give: image 0 -> [90, 100, 110, 0]   image 1 -> [1, 2, 3, 0];")
print(" if you edited example_tiny_net.py, compare with the numbers it printed as expected outputs)")
