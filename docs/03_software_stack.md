# 03 — The Complete Software Stack, Layer by Layer

*An accelerator is never just RTL. Before the chip there is a model, a quantizer, a compiler and a golden reference; around it there is a simulator; after it there is a synthesis flow and a driver. This chapter walks through every layer in the order data flows, tells you what the tool is, why hardware people need it, exactly what to type, and which file in this project plays that role.*

---

## 0. The whole lifecycle in one picture

```
 (1) MODEL            (2) QUANTIZE            (3) COMPILE              (4) RUN
 ───────────          ─────────────           ─────────────           ──────────────────────
 PyTorch  ──train──►  floats → int8  ──────►  tile, pad, lay out ──►  register program
   │                  + bias, scales          weights & descriptors     (prog.hex / C driver)
   ▼                     │                        │                         │
 model.onnx             qnet.npz              prog.hex, net_image.h         ▼
 (the network as a      (integers the         (what the CPU writes     ┌──────────────┐
  file)                  hardware uses)        into the registers)     │ RTL (Verilog)│──► results
                             │                                         └──────────────┘
                             ▼                                                ▲
 (4) GOLDEN REFERENCE  integer-only numpy model ──── must match bit-for-bit ──┘
                        (sw/golden.py)

 (5) VERIFY   Verilator / iverilog / Xcelium / Questa + testbench       (6) IMPLEMENT  Yosys, Genus, Innovus, Tempus, Voltus
 (7) HOST     register map → C driver → AXI4-Lite → CPU / PCIe
```

| # | Layer | Tools (industry) | Used in this project | Files |
|---|---|---|---|---|
| 1 | Model / algorithm | PyTorch, TensorFlow, ONNX | PyTorch + ONNX | `sw/train_mlp.py` → `build/model.onnx` |
| 2 | Quantization | TFLite, Brevitas, ONNX Runtime quantizer, TensorRT | hand-written post-training quantizer | `sw/quantize.py` → `build/qnet.npz` |
| 3 | Compilation | TVM, IREE, FINN, vendor compilers | hand-written compiler, ≈ 285 lines in two files | `sw/compile.py`, `sw/compiler_lib.py` → `build/prog.hex`, `build/net_image.h` |
| 4 | Golden reference | NumPy/C++ model, SystemC/TLM | integer-only NumPy | `sw/golden.py`, `sw/trace_one.py` |
| 5 | RTL simulation & verification | Xcelium, Questa, VCS, Verilator, Icarus, cocotb, UVM, JasperGold | Verilator (primary), Icarus, Verilog testbenches | `tb/`, `scripts/` |
| 6 | Synthesis & implementation | Genus, Design Compiler, Yosys; Innovus/ICC2; Tempus/PrimeTime; Voltus | Yosys for sanity checks; **you** run the Cadence flow | `docs/06_…` |
| 7 | Host driver & control | C/C++ driver, Linux UIO/VFIO, AXI4-Lite, PCIe | C driver + AXI4-Lite bridge | `driver/`, `rtl/axil_reg_bridge.v` |

---

## 1. Model / algorithm layer — PyTorch and ONNX

### What it's for (in hardware terms)

**PyTorch** is a Python library for building and *training* networks. For you it plays exactly one role: it is the **factory that produces the weight constants** and the **specification of the network's structure**. Once training is over, you never need PyTorch again except to compare results.

**ONNX** (Open Neural Network Exchange) is a **file format**: a graph of operations (`Gemm`, `Relu`, `Conv`, …) plus the weight tensors. Think of it as the neutral "netlist" of a neural network — the thing every downstream tool (quantizers, compilers, your hardware compiler) can read regardless of how it was trained. Inspecting our file:

```python
import onnx
m = onnx.load("build/model.onnx")
for n in m.graph.node: print(n.op_type, list(n.input), '->', list(n.output))
for t in m.graph.initializer: print(t.name, list(t.dims))
```
```
Gemm ['x', '0.weight', '0.bias'] -> ['/0/Gemm_output_0']       ← layer 1: x·W1ᵀ + b1
Relu ['/0/Gemm_output_0']        -> ['/1/Relu_output_0']       ← squash
Gemm ['/1/Relu_output_0', '2.weight', '2.bias'] -> ['logits']  ← layer 2
0.weight [16, 64]    0.bias [16]    2.weight [10, 16]    2.bias [10]
```

That is the entire network: three nodes and four tensors. (`Gemm` = general matrix multiply; PyTorch stores each weight as `[out, in]`, so `sw/quantize.py` transposes to the `[K, N]` layout the hardware wants.) To *see* it graphically, open `model.onnx` in the free **Netron** viewer.

### Just enough PyTorch to read `sw/train_mlp.py`

| Line in the file | Plain meaning |
|---|---|
| `nn.Linear(64, 16)` | a layer with a `16×64` weight tensor and a 16-entry bias — a bank of 16 neurons over 64 inputs |
| `nn.ReLU()` | `max(0, x)` |
| `nn.Sequential(...)` | run them in order |
| `loss = CrossEntropyLoss()(model(x), y)` | one number saying how wrong the predictions are (0 = perfect) |
| `loss.backward()` | compute, for every weight, "if I nudge you up, does the error go up or down?" (the *gradient*) — you do not need to know how |
| `opt.step()` (Adam) | nudge every weight a little in the direction that lowers the error |
| loop 300 times | repeat until the error is small |
| `torch.onnx.export(...)` | write the graph + trained weights to `model.onnx` |

Training here takes seconds. Test accuracy after training: **96.94 %** (349 of 360 unseen images — the train/test split is 80/20, and the *test* images are never used for training).

### How you go about using it

```bash
pip install torch onnx onnxruntime numpy scikit-learn      # once
python3 sw/train_mlp.py          # trains, prints accuracy, writes build/model.onnx and build/data.npz
python3 sw/trace_one.py 0 5      # optional: see a single image flow through
```

Things to try (each changes only one line of `train_mlp.py`):

* Widen the hidden layer: `HIDDEN` is an environment variable read by `train_mlp.py` (comma-separated widths), e.g. `HIDDEN=32 HW_ARGS="--acc-depth 64 --wgt-depth 1024" HW_G="-GACC_DEPTH=64 -GWGT_DEPTH=1024" bash scripts/run_all.sh` (the exact commands and results are in `00_START_HERE.md` Step 5). The compiler tells you when a bigger network no longer fits the hardware (`NT_MAX`, `ACC_DEPTH`, weight SRAM depth …) — it reports the first limit it finds, so you may need several rounds.
* Fewer epochs → accuracy drops → quantized accuracy follows it.

**What to remember:** the chip's contract with this layer is *only* the ONNX file (`Gemm` [+ `Relu`] chains). If a future model uses an operation the compiler does not know, `sw/quantize.py` stops and names it.

---

## 2. Quantization layer — floats → integers

### Why it exists

Hardware multiplies integers cheaply (chapter 1 §5). Trained weights are fp32. Quantization is the *offline* process of choosing, for every tensor, an integer representation with a **scale**, such that the network's answers barely change.

### The idea (symmetric per-tensor INT8, what we implemented)

```
real ≈ scale × int         ints are in [-127, 127] for weights, [-128, 127] for activations
```

For layer *ℓ* with input scale `s_x`, weight scale `s_w`, output scale `s_out`:

| Quantity | Formula | Example from our network (layer 1) |
|---|---|---|
| weight scale | `s_w = max|W| / 127` | 0.016849 |
| weight ints | `w_q = round(W / s_w)` | −127 … 127 |
| input scale | `s_x = max|x| / 127` (image pixels 0…1) | 0.007874 (= 1/127) |
| bias ints | `bias_q = round(b / (s_x · s_w))` — int32, lives in the accumulator's units | first four: −543, −1544, 441, −71 |
| output scale | from **calibration**: run the float network on the training set, take the largest output, `s_out = max / 127` | 0.100710 |
| integer accumulator | `acc = Σ x_q · w_q + bias_q` — units of `s_x · s_w` | e.g. 34 207 |
| requantization factor | `M = s_x · s_w / s_out` — a small real number | 0.0013173 |
| hardware constants | `M ≈ mult / 2^shift`, `mult` a 16-bit integer | `mult = 44203`, `shift = 25` |
| INT8 output | `y_q = clamp((acc · mult + 2^(shift−1)) >> shift, lo, 127)` | 34 207 → **45** |

**Why `mult / 2^shift`?** The hardware cannot multiply by 0.0013173, but it can multiply by the integer 44203 and then shift right by 25 (= divide by 33 554 432): 44203 / 2²⁵ = 0.00131735. The relative error of that approximation is about 7 × 10⁻⁶ — negligible. `choose_mult_shift()` in `sw/quantize.py` picks the largest shift for which `mult` still fits in 16 bits.

**Rounding:** adding `2^(shift−1)` before the shift makes it round-to-nearest. Arithmetic (sign-preserving) right shift makes negative numbers round consistently. The RTL and the Python golden model do the identical thing, which is why they agree bit for bit.

**Overflow safety:** the golden model raises an error if any accumulator would exceed the INT32 range (`golden.layer`). Our worst case is far from it: 64 products of two int8 values are at most 64 × 128 × 128 = 2²⁰ ≈ 1 million in magnitude, against a limit of 2³¹ ≈ 2.1 billion.

### PTQ vs QAT

* **Post-training quantization (PTQ)** — quantize an already-trained network. What we do. Fast, needs no retraining, fine for small/easy networks.
* **Quantization-aware training (QAT)** — insert simulated quantization *during* training so the network learns to be robust to it. Needed for harder networks or lower bit-widths (4-bit and below).

### The industry tools and what they add

| Tool | What it is | What it does that our ≈ 120-line quantizer doesn't |
|---|---|---|
| **TensorFlow Lite** (TFLite) | Google's mobile inference format; defines the widely used INT8 quantization spec | per-*channel* (per-neuron) weight scales, asymmetric activations with *zero-points*, fixed-point 32-bit multipliers, operator fusion, a large kernel library |
| **Brevitas** | PyTorch library for quantization-aware training, maintained by AMD (Xilinx Research); exports ONNX with quantization nodes | arbitrary bit-widths (INT4, binary), learned scales; feeds AMD's FINN flow |
| **ONNX Runtime quantizer / TensorRT** | vendor quantization of ONNX models | calibration algorithms (entropy/percentile), per-channel |
| **FINN, hls4ml** | frameworks that turn quantized networks into FPGA/ASIC hardware automatically | generate *the hardware itself* from the network, rather than a fixed accelerator |

What our simplification leaves out and how it would show up: with a single scale per tensor, one large weight forces a coarse scale on all the others; per-channel scales fix that. Zero-points let unsigned data (e.g., post-ReLU) use the full 8 bits. Neither is needed at 96.94 %.

### How you go about using it

```bash
python3 sw/quantize.py
```
prints per-layer scales, the `(mult, shift)` pairs, and the float-vs-INT8 accuracies, and writes `build/qnet.npz` (all integers + scales). If INT8 accuracy is much lower than float, calibrate on more data or move to per-channel scales.

---

## 3. Compilation layer — from integers to a register program

### What a "compiler" means for an accelerator

A CPU compiler turns C into instructions. An **accelerator compiler** turns a *network* into whatever the hardware consumes: here, **the exact sequence of register writes** that loads the weights, biases and layer descriptors, then the inputs. It makes every decision the hardware doesn't:

| Decision | Where in `sw/compiler_lib.py` |
|---|---|
| pad K and N up to multiples of `ROWS` (zero weights) | `pad_layers` |
| cut each weight matrix into `ROWS × COLS` tiles and order them as `p = kt·NT + nt` | `static_ops` |
| assign memory addresses (weight, bias, ping-pong activation halves) | `plan` |
| check that everything fits, naming the violated limit | `plan` (raises with a message) |
| pack int8 lanes into 32-bit register beats | `pack_lanes`, `split_beats` |
| emit layer descriptors | `static_ops` |
| decide readback addresses | `readback_ops` |

Compilers like **TVM** or **IREE** do the same for real chips (plus operator fusion, scheduling, memory planning, convolution lowering, auto-tuning). TVM even ships a reference open-source accelerator, **VTA**, to show the whole path. Ours is the same idea at 1 % of the size.

### Outputs

```
build/prog.hex          one register operation per line: OPCODE(2) 00 ADDR(4) DATA(8)
                        9 677 lines for 45 batches (9 676 register operations + the END marker);
                        the first 568 = probe + load + first batch
build/prog_meta.json    what each READ means (used by sw/report.py)
build/net_image.h       the same network as C arrays (for driver/example_infer.c)
```

```bash
python3 sw/compile.py                    # default: batch 8, whole test set
python3 sw/compile.py --batch 4 --rows 8 --m-max 4 --kt-max 8 ...   # for other hardware sizes
```

The hardware size is *not* stored in the ONNX file; you pass the same parameters to the compiler as you gave the RTL (or accept the defaults). `sw/dnn_cfg.py` is the one place that mirrors the RTL's register map and parameters — if you change a Verilog parameter, change it there.

---

## 4. Architectural & golden-reference simulation

### The rule: never debug arithmetic in the waveform first

You need something that says what the right answer *is* before you write, or while you debug, the RTL. A **golden model** is a slow, obviously-correct, software implementation of the intended behaviour. The verification chain in this project has four rungs, and each one is checked against the one above:

```
  PyTorch float model   ── (onnxruntime) ──► 96.94 %      "is the network any good?"
        │  quantize
        ▼
  Python integer model  ── sw/golden.py ──► 96.94 %      "is INT8 good enough?"  (int64 numpy, overflow-checked)
        │  must match bit for bit
        ▼
  RTL simulation        ── tb_dnn_player ─► 96.94 %      "does the hardware do the same integer maths?"
        │  same register program through AXI / C driver
        ▼
  C driver ≡ Python compiler ─────────────► identical register stream (first batch, byte for byte)
  AXI4-Lite bridge ───────────────────────► same program, same readback (random handshake stalls)
```

`sw/golden.py` is ~30 lines and mirrors `rtl/dnn_ppu.v` step for step (accumulate in int64, add bias, multiply by `mult`, add rounding, arithmetic shift, clamp). Because it uses *integers only*, a mismatch is always a genuine bug — never "floating-point rounding".

### Where SystemC / TLM fits (and why we didn't use it)

**SystemC** is a C++ library for modelling hardware at a higher level than RTL; **TLM-2.0** models buses as function calls. Companies use it to (a) explore architectures long before RTL exists — how many PEs, how big the SRAMs — and (b) give software teams a fast virtual chip to develop drivers on. For this project the Python golden model plays role (a) functionally, and `sw/sweep_arch.py` plays it for performance by running the RTL itself at several sizes (fast enough here). If you later model a multi-core SoC around this block, you would wrap the golden model in a SystemC/TLM peripheral.

### Two reference tools you should try

```bash
python3 sw/trace_one.py 17 5    # every number of image #17 through neuron 5
python3 sw/sweep_arch.py        # 5 array/batch configurations: cycles, utilization, bytes/MAC
```

---

## 5. RTL simulation & verification

### The simulators

| Simulator | Type | Notes |
|---|---|---|
| **Verilator** | open source, compiles RTL to C++; 2-state, very fast | our main tool; `--binary --timing` builds a standalone executable from Verilog testbenches with `#delays` |
| **Icarus Verilog** (`iverilog`) | open source event-driven | slower; used as an independent cross-check in the v1 regression |
| **Cadence Xcelium** (`xrun`) | commercial, 4-state, UVM, coverage | what your Cadence flow would use |
| **Siemens Questa/ModelSim, Synopsys VCS** | commercial | same role |
| **cocotb** | Python testbench framework driving any simulator | an option if you'd rather write tests in Python |

### How this project tests the hardware: a "CPU in a box"

Instead of a bespoke testbench per feature, **one** generic testbench, `tb/tb_dnn_player.v`, reads a *register program* (`prog.hex`) and replays it against the RTL exactly like a driver would — `WRITE`, `WAIT_DONE`, `READ_EXPECT` (must equal the golden value), `READ_LOG`, and so on. Consequences:

* Any workload (the digit network, thousands of random networks, error cases) is just a different `.hex` file.
* The *same* file can be produced by Python or by the C driver — the two are proved identical.
* `+axi_stall` and `-GAXIL=1` send the same program through the AXI4-Lite bridge with randomized handshake delays.

Commands:

```bash
bash scripts/run_all.sh            # train → quantize → compile → simulate → report → C driver → AXI (≈ 30 s; ≈ 25 s with --skip-train)
bash scripts/run_dnn_regress.sh    # 240 random multi-layer networks on 7 hardware configurations + error cases + AXI (≈ 1 min)
bash scripts/run_regress.sh        # v1 matrix-multiply core regression (Verilator + iverilog, ≈ 6 min)
bash scripts/check_driver.sh       # C driver ≡ Python compiler, replayed on RTL
bash scripts/check_rtl_quality.sh  # Verilator -Wall lint + Yosys latch / async-reset scan
bash scripts/check_psum_formula.sh # partial-sum traffic formula == RTL accumulator strobes
bash scripts/run_mutations.sh      # 19 injected bugs must all be caught (≈ 2-3 min)
bash scripts/run_wave.sh           # dumps build/dnn.vcd for gtkwave / Simvision / Verdi
```

**Running the same testbench under Xcelium** (sketch — I could not run Cadence tools here): `xrun -64bit -sv -access +r tb/tb_dnn_player.v rtl/*.v -top tb_dnn_player -defparam tb_dnn_player.AXIL=1 +prog=build/prog.hex +axi_stall`. The testbench is plain Verilog-2001 plus `$urandom`, `$value$plusargs`, `$test$plusargs`.

### How trustworthy are the tests? (mutation testing)

A test that passes proves little unless it would also *fail* on a broken design. So the deliberately-broken-copy technique ("mutation testing") was used: change one thing in the RTL/driver, re-run, confirm the suite goes red. `bash scripts/run_mutations.sh` applies 19 such mutations to copies of the design (7 in the datapath/control RTL, 4 in the error-handling RTL, 4 in the C driver, 4 in the AXI bridge); all 19 are caught (table in `06_verification_and_backend.md`).

### What to add for a professional sign-off

Assertions (SVA), functional coverage, X-propagation runs, gate-level simulation with SDF, and formal checks with JasperGold — listed with concrete properties in `06_…`.

---

## 6. Synthesis & implementation (EDA)

### What synthesis actually does

RTL says *what* registers exist and what logic feeds them. **Logic synthesis** turns that into a netlist of cells from a technology library: parse RTL → infer flip-flops/adders/multipliers/muxes → optimize Boolean logic → map to the library's cells to meet **timing** (from your SDC constraints) with the smallest **area**/**power**. Then place-and-route (Innovus) places the cells, builds the clock tree and wires them; static timing analysis (Tempus) checks every path; power analysis (Voltus/Joules) estimates energy.

| Tool | Company | Role |
|---|---|---|
| **Yosys** | open source | RTL → generic gates; I used it only for *structural sanity checks*: no latches inferred, the only asynchronous element is the reset synchronizer, hierarchy elaborates |
| **Genus** | Cadence | production synthesis |
| **Design Compiler** | Synopsys | equivalent role |
| **Innovus / ICC2** | Cadence / Synopsys | place & route |
| **Tempus / PrimeTime** | Cadence / Synopsys | static timing |
| **Voltus / Joules / PrimePower** | Cadence / Synopsys | power |
| **Conformal / Formality** | Cadence / Synopsys | RTL-vs-netlist equivalence |

You said you will drive this flow yourself, so `06_verification_and_backend.md` contains only what is *specific to this design*: memory macros, reset/clock structure, SDC skeleton, likely critical paths, and how to get the missing energy number by feeding **switching activity from these very simulations** into Genus/Voltus/Joules.

### Why it matters that the design is synthesizable Verilog-2001

No `initial` blocks that synthesis would see (three exist — in `dnn_accel_core.v`, `lrb.v` and the v1 top `systolic_accel_top.v` — but they are only parameter sanity checks using `$display`/`$finish`, wrapped in `// synopsys translate_off … translate_on`; Yosys honours the pragma and Genus supports the same one, though I could not run Genus), no delays, no latches, one clock, synchronous reset everywhere except the documented reset synchronizer, `$clog2` for widths only. Verilator `-Wall` reports zero warnings on both tops (`dnn_accel_top`, `dnn_accel_axil_top`); a few intentional width/unused-signal waivers are documented in the file headers.

---

## 7. Host driver & control interface — how a CPU talks to the accelerator

### The mental model

To software, the accelerator is **a block of memory**. A load from address `BASE + 4·n` reads register `n`; a store writes it. There is no other channel. Everything — inserting weights, describing layers, starting, waiting, reading results — is a sequence of loads and stores. The **driver** is the C code that performs the right sequence. The **register map** (`05_register_map_and_driver.md`) is the contract.

### The bus in between: AXI4-Lite

Most SoCs (ARM, RISC-V with AXI fabrics, Zynq/FPGA designs) connect peripherals with **AXI4-Lite**: five small channels, each with a `VALID`/`READY` handshake (a transfer happens on a clock edge where both are high):

```
 write:  AW (address) + W (data)  ───►   B (response)
 read :  AR (address)             ───►   R (data + response)
```

`rtl/axil_reg_bridge.v` (about 120 lines) converts that into the accelerator's simple register bus (`reg_wr_en`, `reg_rd_en`, `reg_addr`, `reg_wdata`, `reg_rdata`); `rtl/dnn_accel_axil_top.v` wraps it with the accelerator. Byte address = 4 × word address. Its testbench master drives AW and W **independently**, with random delays, and delays `BREADY`/`RREADY` — protocol corner cases where naive bridges fail — and the RTL still returns identical results.

### PCIe, briefly

For a plug-in card the host reaches the same registers through a **PCIe BAR** (a window of the card's address space mapped into the PC's memory space). In Linux, a kernel driver, or a user-space mapping (UIO/VFIO), turns register accesses into loads and stores to that window; bulk data (weights, images) normally moves by **DMA** — the accelerator (or a DMA engine) reads/writes host memory itself over AXI4 (full) or AXI-Stream. This project moves data through the register windows (simple, slow) because the payload is a few hundred bytes per inference; DMA is the natural next step for bigger models. The driver API would be unchanged.

### The C driver in this project

```
driver/dnn_accel.h     the API and register constants
driver/dnn_accel.c     seven functions: probe, load_network, write_input, run, read_perf, read_output, argmax
driver/hal_mmio.h      the tiny HAL for real memory-mapped hardware
driver/example_infer.c full inference using a *recording* HAL
```

The driver never touches an address directly; it calls `hal->read()` / `hal->write()`. On a real SoC the HAL is just `base[word_addr] = data` on a `volatile` pointer (`hal_mmio.h`, a dozen lines). Here it is a **recorder** that writes each access to a file; `scripts/check_driver.sh` then proves (1) the C driver's stream equals the Python compiler's stream byte for byte (568 register accesses), and (2) replayed on the RTL, it passes. Full details, sequence diagrams and a porting recipe in **`05_register_map_and_driver.md`**.

---

## 8. Which tool do I open when …?

| I want to … | Open / run |
|---|---|
| understand what the network is | `python3 sw/trace_one.py`, Netron on `build/model.onnx` |
| change the network | `sw/train_mlp.py` → `bash scripts/run_all.sh` |
| see why INT8 accuracy dropped | `sw/quantize.py` output (scales), `sw/golden.py` |
| find out why a register program is rejected | error text from `sw/compile.py` (names the limit); `STATUS.err` on the chip |
| debug the hardware | `bash scripts/run_wave.sh` → gtkwave `build/dnn.vcd` |
| compare array sizes | `python3 sw/sweep_arch.py` |
| write firmware | `driver/dnn_accel.h`, `driver/hal_mmio.h` |
| synthesize | `docs/06_verification_and_backend.md` §3–§5 |
