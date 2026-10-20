# Systolic-Array INT8 DNN Accelerator

Weight-stationary INT8 systolic array with a local reuse buffer, a layer sequencer, a bias/requantize/ReLU unit, a register interface (+ AXI4-Lite bridge), a Python software flow (PyTorch → ONNX → INT8 → register program), and a C host driver. A trained network runs on the RTL and matches the integer reference bit for bit.

```bash
pip install -r requirements.txt      # torch onnx onnxruntime numpy scikit-learn
                                     # also needed: verilator >= 5 (tested 5.020), gcc, make, g++ >= 10
bash scripts/run_all.sh              # train → quantize → compile → simulate → report → C driver → AXI  (~30 s)
```

**Cadence / ModelSim users (no Verilator): open `docs/BEGINNER_GUIDE.html` in a browser first.** It has click-by-click steps; the launch files are in `cadence/` (`bash cadence/run_xcelium.sh`, or `cadence\run_modelsim.bat` on Windows). The `scripts/*.sh` files need Verilator and are optional. To understand what each RTL file and each testbench does (with diagrams and what to expect on screen), run `python sw\compare_one.py` to compare the chip with Python score by score; run `python sw\show_digits.py` to see the actual 8x8 pictures in your browser; open `docs/RTL_EXPLAINED.html`; the single-PE testbench runs with `bash cadence/run_pe_xcelium.sh` or `cadence\run_modelsim.bat pe`.

**Then read `docs/00_START_HERE.md`** — it answers "what is done, is it efficient, and what do I do next", and links the rest of the documentation (`01`–`06`).

| Folder | Contents |
|---|---|
| `rtl/` | synthesizable Verilog-2001 (top: `dnn_accel_top` or `dnn_accel_axil_top`) |
| `sw/` | train / quantize / compile / golden / report / learning tools |
| `driver/` | C99 host driver, memory-mapped HAL, example |
| `cadence/` | launch files for Xcelium, ModelSim, Python (Windows) and Genus, plus the SDC |
| `tb/`, `scripts/` | testbenches and one-command flows/regressions (scripts need Verilator) |
| `docs/` | documentation |
| `build/` | generated artifacts (model.onnx, qnet.npz, prog.hex, net_image.h …) |
