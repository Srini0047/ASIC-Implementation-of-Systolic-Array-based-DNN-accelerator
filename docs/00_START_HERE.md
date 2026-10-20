# 00 — START HERE

**Project:** ASIC implementation of a systolic-array based INT8 DNN accelerator — RTL, software flow, C driver, AXI4-Lite bridge, tests and documentation.

**Status of this delivery (v2):** a complete, simulated, bit-exact-verified inference pipeline for fully-connected networks. **Not done:** synthesis, timing, power and area numbers (those need your Cadence flow), convolution layers.

> **New here, or unsure what to click and type? Open `docs/BEGINNER_GUIDE.html` in a web browser first.** It explains from zero what every folder is for (including `sw/`, `driver/`, `scripts/`, `build/`) and gives exact steps for Xcelium, ModelSim, Python and Genus. The launch files for Cadence tools are in `cadence/`. To understand the chip itself, open `docs/RTL_EXPLAINED.html`: it walks through all 15 RTL files and both testbenches (`tb_pe`, `tb_dnn_player`) with diagrams and real waveforms, and says what the simulation output should look like. This file (00) and the chapters 01–06 are the reference material behind them.

---

## 1. What you have

```
systolic_dnn_accel/
├── README.md                    quick start (install, one command, where to read next)
├── requirements.txt             Python packages
├── cadence/                     launch files for YOUR tools: run_xcelium.sh, run_pe_xcelium.sh, run_modelsim.bat/.do, python_*.bat, genus_synth.tcl, constraints.sdc
├── docs/
│   ├── BEGINNER_GUIDE.html      click-by-click guide for Xcelium / ModelSim / Python / Genus (open in a browser)
│   ├── RTL_EXPLAINED.html       the 15 RTL files and the 2 testbenches explained with diagrams and waveforms (open in a browser)
│   ├── 00_START_HERE.md         ← this file: answers + your job list
│   ├── 01_DNN_for_hardware_engineers.md   DNN basics in RTL vocabulary, energy/roofline, CPU/GPU comparison
│   ├── 02_project_mapping.md    what you are building, block by block; tiling; cycle-level walk; array-size study
│   ├── 03_software_stack.md     PyTorch/ONNX → quantization → compiler → golden → simulators → synthesis → driver
│   ├── 04_roadmap.md            phased micro-milestones (your Step 1 → Step N), extensions, demo fast-track
│   ├── 05_register_map_and_driver.md      the HW/SW contract, programming sequences, C driver, AXI4-Lite
│   ├── 06_verification_and_backend.md     what was tested (and not), assertions, SDC skeleton, memory macros, power flow
│   └── DESIGN_NOTES.md          v1 (matmul core) design notes: Phase 1 analysis, code evaluation, Phase 3 handoff
├── rtl/                         Verilog-2001, synthesizable
│   ├── pe.v  skew_delay.v  systolic_array.v  sa_core.v  sram_sdp.v  lrb.v  rst_sync.v  perf_counters.v      (v1 core)
│   ├── dnn_ppu.v  dnn_ctrl.v  dnn_accel_core.v  dnn_reg_if.v  dnn_accel_top.v                                (v2 DNN engine)
│   ├── axil_reg_bridge.v  dnn_accel_axil_top.v                                                               (AXI4-Lite SoC wrapper)
│   └── accel_ctrl.v  systolic_accel_top.v                                                                    (v1 matmul-only top)
├── sw/                          Python: train, quantize, compile, golden, report, sweep_arch.py, trace_one.py, learning tools
├── driver/                      C host driver (portable C99) + memory-mapped HAL + example
├── tb/                          testbenches (generic register-program player, PE, v1 core)
├── scripts/                     run_all.sh · run_dnn_regress.sh · run_regress.sh (v1 core) · check_driver.sh ·
│                                check_rtl_quality.sh (lint + Yosys) · check_psum_formula.sh · run_mutations.sh ·
│                                run_wave.sh · run_tiny.sh
└── build/                       generated artifacts (model.onnx, qnet.npz, prog.hex, net_image.h, …)
```

---

## 2. The direct answers to your questions

### 2.1 "Have the project files you already gave me solved all these issues?"

**No — the first delivery (v1) was a matrix-multiply engine, not yet a neural-network accelerator.** It had the right *heart* (INT8 weight-stationary systolic array + reuse buffer) but could not run a network. Here is the honest status of each thing a DNN accelerator has to do:

| # | What a DNN accelerator needs | v1 (first delivery) | v2 (now) |
|---|---|---|---|
| 1 | INT8 MAC array with weight reuse | ✅ | ✅ (unchanged, verified) |
| 2 | Activation reuse (Local Reuse Buffer) | ✅ | ✅ (3.8× fewer activation-SRAM reads, measured) |
| 3 | **Layers with K larger than the array** (accumulate across tiles) | ❌ (K fixed = ROWS) | ✅ accumulator SRAM, read-modify-write |
| 4 | **Bias + requantize INT32→INT8 + ReLU** | ❌ | ✅ `dnn_ppu.v` |
| 5 | **Several layers back-to-back without the host** | ❌ | ✅ descriptors + ping-pong activation regions |
| 6 | **A real trained network on it** | ❌ | ✅ PyTorch digit MLP, 96.94 % (RTL = INT8 golden = float) |
| 7 | **Software flow** (PyTorch → ONNX → INT8 → program) | ❌ | ✅ `sw/` |
| 8 | **Host driver** | ❌ | ✅ C99 driver, proven identical to the Python compiler |
| 9 | **SoC bus interface** | ❌ | ✅ register bus + AXI4-Lite bridge (stall-tested) |
| 10 | Explanations for an RTL engineer with no ML background | ❌ | ✅ this documentation set |
| 11 | Convolution layers | ❌ | ❌ **still missing** (roadmap 7.5: im2col in software) |
| 12 | Overlap of weight load / compute / drain | ❌ | ❌ **still missing** (roadmap 7.1, 7.2) |
| 13 | Area / timing / power / pJ per inference | ❌ | ❌ **needs your Cadence flow** (`06_…` §3.4) |

### 2.2 "And is it more efficient now?"

*Efficient* can mean four different things; the honest answer differs for each. Everything marked "measured" comes from running the RTL (default 4×4 array, batch of 8, 2-layer digit network); "derived" is arithmetic on measured counts; "unknown" needs your tools.

| Dimension | Answer | Evidence |
|---|---|---|
| **Speed (cycles)** | ✅ Yes — about **5.35× fewer cycles** than a design doing one MAC per cycle | **measured:** 221 cycles/image; 1 184 MACs/image → 5.35 useful MAC/cycle of a 16 peak (33.4 % utilization) |
| **Weight fetches** | ✅ ≈ 7.8× fewer weight-SRAM bytes than fetching a weight per MAC (each weight is used by all 8 images in the batch) | **measured:** 304 weight-SRAM words (1 216 B) per 9 472 useful MACs = 0.128 B/MAC, versus 1 B/MAC without reuse |
| **Activation fetches** | ✅ Yes — the reuse buffer cuts activation-SRAM reads **3.8×** | **measured:** 160 SRAM reads + 448 LRB reads instead of 608 SRAM reads |
| **Host involvement** | ✅ Yes — 32-bit partial sums and intermediate activations never leave the chip: host moves ≈ 76 bytes/image (64 in, 12 out) instead of the ≈ 1 216 bytes/image of partial sums the v1 core would have exported | **derived** from tile geometry (76 passes × 128 B ÷ 8 images) |
| **Total on-chip bytes moved per MAC** | ❌ **Not better at 4×4** — 2.25 B/MAC versus 2.0 for a design with no reuse at all, because partial sums are spilled to the accumulator SRAM 15 times per output (K = 64 needs 16 K-tiles) | **measured** counters + geometry; the partial-sum formula is checked against the RTL's real accumulator-SRAM strobes for the default run (`scripts/check_psum_formula.sh`) and for all 5 sweep configurations (`sw/sweep_arch.py` fails if they differ) |
| … at a larger array | ✅ 8×8: 1.03–1.14 · 16×16: **0.43 B/MAC** (4.7× better than no reuse) | `python3 sw/sweep_arch.py`, chapter 02 §8 |
| **Energy (pJ/MAC), area, fmax** | ❓ **Unknown** — not measured | no synthesis run; see `06_…` §3.4 for how you get it |

**The one-paragraph verdict.** The design is now a *system* that runs an entire quantized network on-chip, on its own, in ≈ 5× fewer cycles than a serial MAC unit, with the expected reuse of weights and activations. It is **not** yet a demonstrated energy win, and at the default 4×4 size the partial-sum traffic cancels the operand-reuse savings in raw bytes moved; a larger array or an output-stationary variant fixes that (roadmap 7.3, 7.4), and Genus/Voltus will tell you what it is worth in picojoules. Also, only 34 % of cycles are the array streaming data — loading weights (17 %) and draining the pipeline (48 %) are not yet overlapped (roadmap 7.1, 7.2).

### 2.3 How this addresses the CPU/GPU problems from chapter 01

| Problem with CPUs/GPUs | This design |
|---|---|
| Every MAC pays instruction fetch/decode | no instructions: a small FSM sequences tiles ✅ |
| Operands come from register files/caches | operands come from the neighbouring PE's register ✅ |
| fp32 arithmetic and 4× the bytes | INT8 multiply, INT32 accumulate ✅ (≈ 15× less arithmetic energy by the Horowitz table) |
| Weights re-fetched for every input | weight-stationary + batch ✅ (≈ 7.8×) |
| Activations re-fetched for every output neuron | reuse buffer ✅ (3.8×) |
| Partial sums shuttled through memory | ⚠️ only partly: neighbour-to-neighbour inside a tile, but SRAM read-modify-write across K-tiles |

---

## 3. What YOU need to do — step by step

### Step 1 — Set up (10 min)

```bash
unzip systolic_dnn_accel_v2.zip && cd systolic_dnn_accel
pip install -r requirements.txt            # torch onnx onnxruntime numpy scikit-learn
verilator --version                        # need >= 5.0 (built and tested with 5.020)
```
Also needed: `gcc`, `make` and `g++` (Verilator's `--binary --timing` generates and compiles C++ with coroutines, so use g++ ≥ 10), `python3` ≥ 3.10. Optional: `iverilog` (extra cross-check in `run_regress.sh`), `yosys` (`check_rtl_quality.sh`), `gtkwave`.

(If you received the project as a folder rather than the zip, skip the `unzip`.)

### Step 2 — Run everything once (about 30 s on the machine used here; more the first time, when PyTorch trains)

```bash
bash scripts/run_all.sh
```
You should see the following **key lines** (this is a condensed extract, not the literal output — the real run prints more, e.g. per-layer quantization scales and the full report table; the numbers are deterministic):
```
float (onnxruntime) accuracy : 96.94%
INT8 integer-only accuracy   : 96.94%
PLAYER: 9677 ops, 1313 reads, 0 mismatches, 0 timeouts, ...      *** PLAYER PASSED ***
INT8 accuracy measured FROM RTL  :  96.94 %
busy cycles : 1771 (221 cycles/image)   useful MACs : 9472 (5.35 MAC/cycle, peak 16)   array utilization 33.4 %
IDENTICAL: 568 register accesses (C driver == Python compiler)   ...   DRIVER CHECK PASSED
AXI4-Lite readback identical to register-bus readback
ALL STEPS DONE
```
(`PLAYER` lines appear three times — the main run, the driver replay and the AXI run; the "ops/reads" numbers of the driver replay are 569/37.) If a step fails the script stops there and prints the tail of that step's log; a missing tool shows up as "command not found" in the first failing step.

### Step 3 — Learn the ideas (a few hours of reading, in this order)

1. `docs/01_DNN_for_hardware_engineers.md` — what a DNN is; why memory movement matters.
2. `docs/02_project_mapping.md` — what each block does; how a 64×16 layer runs on a 4×4 array.
3. `docs/03_software_stack.md` — the tools; skim §1–§4 now, §5–§7 when you need them.

While reading, keep these open in a second terminal:

```bash
python3 sw/trace_one.py 0 5       # one image, every number (chapter 01 §3)
bash scripts/run_tiny.sh          # 4→4 layer you can compute with a pencil
bash scripts/run_wave.sh          # then: gtkwave build/dnn.vcd
python3 sw/sweep_arch.py          # 5 array/batch sizes (≈ 45 s) — also checks the partial-sum formula against the RTL
```

### Step 4 — Build it yourself (this is where you learn the most)

Follow `docs/04_roadmap.md` phases 1–3: write the PE, the 2×2 array, the array with skew, the SRAM/FSM, the reuse buffer, the ping-pong regions, the accumulate stage and the post-processing unit **from the specs given**, compare with the reference in `rtl/`, and use the provided tests. Do at least 1.1, 1.3, 2.5, 3.1, 3.2 — those five carry the ideas that make this a DNN accelerator instead of a matrix multiplier.

### Step 5 — Make the network your own (30 min)

```bash
HIDDEN=32 HW_ARGS="--acc-depth 64 --wgt-depth 1024" \
HW_G="-GACC_DEPTH=64 -GWGT_DEPTH=1024" bash scripts/run_all.sh          # 64→32→10
HIDDEN=32,16 HW_ARGS="--acc-depth 64 --wgt-depth 1024" \
HW_G="-GACC_DEPTH=64 -GWGT_DEPTH=1024" bash scripts/run_all.sh          # 64→32→16→10 (3 layers)
```
Both were run and are bit-exact (INT8 accuracy 96.67 % and 95.83 %). Then restore defaults with `bash scripts/run_all.sh` (without the variables). `HIDDEN` is an environment variable read by `sw/train_mlp.py` (comma-separated hidden widths). If a network does not fit, `sw/compile.py` stops with an error naming the first hardware limit that is exceeded — raise it with `HW_ARGS`, and **you must give the same value to the RTL parameters** (`HW_G` does it in the scripts). Example: `HIDDEN=64` fails with `layer 0: NT > NT_MAX` (raise `--nt-max` and the matching `-GNT_MAX`), and further limits (`ACC_DEPTH`, `WGT_DEPTH`, …) may then report in turn.

### Step 6 — Pick an extension (optional but recommended for the "AI title")

Roadmap Phase 7. The two with best impact per effort: **7.3** (array/batch study — already scripted, S) and **7.1** (overlap weight load with compute — M). Both have measured baselines to compare against.

### Step 7 — Your back-end flow

Read `docs/06_verification_and_backend.md`: memory macros, SDC skeleton, reset and clock facts, likely critical paths, and — importantly — how to turn the RTL simulation into **pJ per inference** with Voltus/Joules. That is the number that turns the qualitative efficiency claims above into evidence.

### Step 8 — Report / presentation

Use `docs/04_roadmap.md` §8 (fast track). The story in one line: *"Weight-stationary INT8 systolic accelerator; a complete PyTorch-trained network runs on-chip; RTL is bit-exact to an integer reference (96.94 %); 5.35 MAC/cycle; here's what limits it and how I'd fix it."*

---

## 4. What changed since the first delivery

**Modified:** `sa_core.v` (new output `res_vld_early`, needed for K-tile accumulation), `systolic_accel_top.v` (ties it off), `DESIGN_NOTES.md` (banner pointing to this set).

**New RTL:** `dnn_ppu.v`, `dnn_ctrl.v`, `dnn_accel_core.v`, `dnn_reg_if.v`, `dnn_accel_top.v`, `axil_reg_bridge.v`, `dnn_accel_axil_top.v`.

**New software:** `sw/` (train, quantize, golden, compiler, report, sweep, trace, tiny example), `driver/` (C driver + HAL + example).

**New tests/scripts:** `tb/tb_dnn_player.v`, `scripts/run_all.sh`, `run_dnn_regress.sh`, `check_driver.sh`, `check_rtl_quality.sh`, `check_psum_formula.sh`, `run_mutations.sh`, `run_wave.sh`, `run_tiny.sh` (the v1 regression `run_regress.sh` was delivered earlier).

**New documentation:** this set (`00`–`06`).

**Bugs found and fixed while building v2** (so you know what kinds of mistakes to look for): an unsigned-ternary bug in the PPU clamp; a 1-bit implicit net from a signal used before its declaration; a `[(D-1)*W-1:0]` slice that is illegal for depth 1 in `skew_delay`; an undeclared-width localparam in a port of `lrb`; ONNX export fixed at batch 1.

**Found by an independent review of the finished package and fixed:** (1) a stale `err` flag could be read as set for a couple of cycles right after START of the *next* run (STATUS now masks `err` like `done`, and the driver polls `busy` first); (2) an out-of-range `N_LAYERS` write wrapped around in the 3-bit register (values such as 9 became 1) — it now reads back 0, which the controller rejects; (3) the regression did not exercise those two cases or the `M·NT > ACC_DEPTH` check, so it now does; (4) several documentation cross-references and numbers.

---

## 5. Limitations, stated plainly

* **No Cadence results.** Nothing here was run through Genus, Innovus, Tempus or Voltus. Area, frequency, power and pJ/MAC are unknown.
* **Fully-connected layers only.** Convolution needs im2col (software) or new hardware; pooling and batch-norm are not implemented.
* **Only `Gemm` (+`Relu`) ONNX models.** The quantizer stops with a message on anything else.
* **Utilization 33 %** at batch 8 (34 % of cycles are the array streaming; the rest is load and drain).
* **Behavioural SRAMs.** If synthesized as written the five SRAMs (30 976 bits at the defaults) become flip-flops; use macros.
* **`ROWS` must equal `COLS`** (output tile = next layer's input slice).
* **Single clock domain.** No CDC logic for a slower/faster SoC bus.
* **Symmetric per-tensor INT8** quantization; fine for this network, may lose accuracy on harder ones.
* **Driver not run on real hardware** (there is none). It is proven against the Python compiler and the RTL through the recorder HAL.
* **Xcelium/Questa untested.** Everything ran on Verilator (and Icarus for v1).

---

## 6. Where to look when …

| Question | Look at |
|---|---|
| "What is a tensor / MAC / quantization?" | `01` §2, §4, §8 |
| "What is this block for?" | `02` §2 |
| "Why does a 64×16 layer take 64 passes?" | `02` §4 |
| "Which tool do I use for X?" | `03` §0, §8 |
| "What are the next 10 things to do?" | `04` |
| "Which register do I write for Y?" | `05` §2 |
| "Was Z verified?" | `06` §1 |
| "What did the original report's code get wrong?" | `DESIGN_NOTES.md` §2 |
