# 04 — Step-by-Step Learning and Implementation Roadmap

*From "I can write Verilog from a spec" to "I built, verified and can explain a working neural-network accelerator." Each step is a micro-milestone with: the goal, what you do, the reference file to compare against, the command that checks it, and a "done when" condition.*

How to use it: for phases 1–3 **write the RTL yourself from the spec given**, then compare with the reference file — that is your strength and the fastest way to internalize the architecture. Every reference module in `rtl/` already passes all tests, so you can also skip a step by reading the reference and jumping ahead.

Size guide: **S** = under an hour, **M** = an evening, **L** = a few evenings.

```
Phase 0  run it, see it                       (S)
Phase 1  the MAC and the array                (M)   ← you write RTL
Phase 2  memories, controller, reuse, ping-pong (L)  ← you write RTL
Phase 3  turn a matmul core into a DNN engine (L)   ← you write RTL
Phase 4  the software flow                    (M)
Phase 5  host: C driver and AXI4-Lite         (M)
Phase 6  ASIC preparation and numbers         (yours)
Phase 7  extensions that add "title impact"   (pick 1–2)
```

**Fast track for a demo** (if time is short): Phase 0 → read chapters 01–02 → run `trace_one.py`, `sweep_arch.py`, `run_wave.sh` → Phase 6 numbers. See §8.

---

## Phase 0 — Run it and look at it (S)

| Step | Do | Done when |
|---|---|---|
| **0.1** Install | Python ≥ 3.10; `pip install -r requirements.txt` (torch ≥ 2.5, onnx, onnxruntime, numpy, scikit-learn); Verilator ≥ 5.0 (tested with 5.020); `gcc`, `make` and `g++` ≥ 10 (Verilator builds C++ with coroutines for `--timing`). Optional: Icarus Verilog, Yosys, GTKWave. | `verilator --version` and `python3 -c "import torch, onnx"` work |
| **0.2** Run everything | `bash scripts/run_all.sh` | last line `ALL STEPS DONE`; report shows `INT8 accuracy measured FROM RTL : 96.94 %` |
| **0.3** Follow one image | `python3 sw/trace_one.py 0 5` | you can explain every line of the printout (chapter 01 §3) |
| **0.4** Open a waveform | `bash scripts/run_wave.sh`, then `gtkwave build/dnn.vcd` | you have found `u_core.u_ctrl.state` cycling LDW → COMP → DRAIN |
| **0.5** Hand-checkable run | `bash scripts/run_tiny.sh` | RTL returns lanes `[90, 100, 110, 0]` |

---

## Phase 1 — The MAC and the array (M)

**1.1 — A signed 8-bit MAC processing element.**
*Spec.* Ports: `clk`, `rst_n` (sync, active-low), `en`, `weight_load`, `weight_in`, `act_in` (signed 8), `psum_in` (signed 32); outputs `weight_out`, `act_out`, `psum_out`. Each clock with `en`: `act_out <= act_in; psum_out <= psum_in + weight_reg*act_in; if (weight_load) weight_reg <= weight_in;` `weight_out` is `weight_reg`.
*Reference.* `rtl/pe.v`. *Check.* copy `tb/tb_pe.v`, change the module name. *Done when* weight −3 × activation 7 gives **−21** (the classic bug: unsigned regs give 1771, which is what the original report's PE did).
*Pitfall to internalize:* `signed` must be on the ports, the register **and** the product; a single unsigned operand in an expression makes the whole expression unsigned.

**1.2 — One MAC, four cycles (optional, 20 min).**
Compute `y = Σ_{i<4} x[i]·w[i]` with one PE, a counter and an accumulator register: 4-cycle latency, 1/4 result per cycle. *Learning:* this is a CPU-style datapath; the array exists to turn "4 cycles" into "1 cycle per result".

**1.3 — 2×2 array by hand.**
Instantiate four PEs; activations move right, partial sums move down; row 1's activation gets **one flip-flop of delay** (skew). Preload `W = [[5,6],[7,8]]`, feed vectors `[1,2]` then `[3,4]`. *Expected:* `[19,22]` then `[43,50]`, each result appearing ROWS + COLS − 1 = 3 cycles after its input vector enters.
*Reference.* `tb/trace_2x2.v` prints the full cycle-by-cycle table (also in `DESIGN_NOTES.md` §3.4). *Done when* your waveform matches that table cycle for cycle.

**1.4 — N×N with generate loops + skew/deskew.**
Parameterize by `ROWS`, `COLS`; add `skew_delay` (row *r* delayed *r* cycles) at the input and the mirror-image deskew (column *c* delayed *COLS−1−c*) at the output, plus a `valid` shift register of depth `ROWS+COLS−1`.
*Reference.* `rtl/systolic_array.v`, `rtl/skew_delay.v`, `rtl/sa_core.v`.
*Check.* write a testbench that instantiates **your core and `sa_core` side by side** and drives both with random vectors: outputs must be identical every cycle. *Done when* 10 000 random vectors match at 2×2, 4×4, 3×5.
*Latency rule to confirm:* a vector entering at cycle τ produces aligned results at τ + ROWS + COLS − 1.

**1.5 — Weight loading.**
Add the shift-chain load: while `weight_load` the weight registers shift **down** each column. Consequence to derive yourself: the word fed *first* ends in the *bottom* row, so the controller must feed tile row ROWS−1 first. *Done when* your array computes `A·W` for a random `W` loaded through `wgt_in`.

---

## Phase 2 — Memories, controller, reuse, ping-pong (L)

**2.1 — The memory model.**
A 1-read/1-write synchronous memory with **registered read data** (`rd_data` appears one cycle after `rd_en`). *Reference.* `rtl/sram_sdp.v`. *Done when* a testbench shows write → read-back latency of exactly one cycle. *Why it matters:* every timing constant in the controller assumes it, and real SRAM macros behave this way.

**2.2 — Controller for one weight tile (v1 FSM).**
States `IDLE → LDW (ROWS cycles) → COMP (M cycles) → DRAIN → IDLE`. Draw the state diagram first, with what each state drives (read enables, addresses, `wgt_load`, `feed_vld`). *Reference.* `rtl/accel_ctrl.v`. *Check.* `bash scripts/run_regress.sh` uses `tb/tb_systolic_accel.v`; it compares results with an in-testbench golden model.

**2.3 — Tile over N and predict the cycle count.**
Repeat the FSM for `T` weight tiles to cover `N = T·COLS` outputs. Before simulating, compute the cycles: `T·(M + 2·ROWS + COLS + 2)`; the testbench checks the formula exactly. *Done when* your prediction and the measured `PERF_CYCLES` agree.

**2.4 — Local Reuse Buffer.**
Small memory next to the array: on the first N-tile, the vectors read from the big activation SRAM are also **copied** into the LRB; on later tiles they are **replayed** from it. *Predict:* activation-SRAM reads fall from `T·M` to `M` (when `M ≤ LRB_DEPTH`). *Reference.* `rtl/lrb.v`. *Check:* `PERF_ACT_RD` and `PERF_LRB_RD` (register `0x21`, `0x22`). *Done when* at `M = 8`, `T = 4` you read 8 activation words + 24 from the LRB (this is the worked example in `DESIGN_NOTES.md` §3.5).

**2.5 — Ping-pong buffer management (your "Step 3").**
*Problem.* Layer *i* produces outputs that layer *i+1* consumes. If both use the same buffer, layer *i+1* would overwrite inputs it has not read yet.
*Solution.* Split the activation SRAM in two regions A and B. Layer 0 reads A and writes B; layer 1 reads B and writes A; and so on. No copying: only the base addresses in the descriptors change (`SRC`, `DST`).
*Spec.* `SRC_i = (i even) ? 0 : BANK`, `DST_i = (i even) ? BANK : 0`, `BANK = M_MAX·KT_MAX`. The output layout `Y[m][nt] at DST + m·NT + nt` is exactly the next layer's input layout `X[m][kt] at SRC + m·KT + kt` (with `KT_next = NT_this`) — which is why `ROWS == COLS` is required.
*Reference.* `sw/compiler_lib.py` (`plan`), `rtl/dnn_ctrl.v` (`act_addr`, `dst_start`). *Done when* a two-layer network runs from one START and the second layer's result matches `sw/golden.py` (`bash scripts/run_all.sh`; layer addresses printed by `sw/compile.py`: `src=0 dst=128`, then `src=128 dst=0`).

---

## Phase 3 — From matmul core to DNN engine (L)

**3.1 — Accumulate across K-tiles.**
*Problem.* The array only sums 4 inputs at a time (`K = ROWS`). A layer with `K = 64` needs 16 passes whose results must be **added**.
*Solution.* Accumulator SRAM. Pass `kt = 0` writes results; later passes read the old partial sum, add, write back; the last pass sends the total onward.
*Trap.* The accumulator SRAM has 1-cycle read latency, so the read must start **before** the result appears: `sa_core` exposes `res_vld_early` (valid pipeline tapped one stage earlier). Draw the timing diagram before coding.
*Reference.* `rtl/dnn_accel_core.v` (accumulate section). *Check.* random networks with `KT` from 1 to 16 in `run_dnn_regress.sh`. *Done when* you can explain why `acc_rd_en = res_vld_early & ~first_kt`.

**3.2 — Post-processing: bias, requantize, ReLU.**
*Spec.* `y = clamp(((acc + bias)·mult + round) >> shift, lo, 127)`, `lo = relu ? 0 : −128`, `round = shift ? 1 << (shift−1) : 0`. Three registered stages, four lanes.
*Reference.* `rtl/dnn_ppu.v`; software twin `sw/golden.py::requant`. *Check.* feed both the same random `(acc, mult, shift, relu)` values — use `run_dnn_regress.sh` with a tiny `M_MAX`. *Corner cases:* `shift = 0`, `shift = 47`, exactly-at-saturation, negative numbers (arithmetic shift!), ReLU with negative input.
*Trap.* mixing an unsigned concatenation with a signed literal in one ternary makes the whole result unsigned. That was an actual bug found while building this design: `clamp_lo` came out unsigned and broke the negative clamp; the fix keeps both branches signed (`relu ? 0 : -128`).

**3.3 — Descriptors and the layer sequencer.**
A table of layer descriptors (the "instruction memory" of a very small machine) and an FSM that loops `layer → kt → nt → pass`. Add validation (`M`, `KT`, `NT`, `M·NT ≤ ACC_DEPTH`, `SHIFT ≤ 47`, and `N_LAYERS` in range) with an `err` flag. Remember that each layer is validated when *it* starts, not at START. *Reference.* `rtl/dnn_ctrl.v`. *Done when* the negative tests in `run_dnn_regress.sh` (`WAIT_DONE_ERR`) pass — they cover every check listed here.

**3.4 — Register interface.**
Address decode, beat staging for wide words (commit on the last beat), `STATUS` with race-free `busy/done` on START, read-back handshake (`rd_valid`). *Reference.* `rtl/dnn_reg_if.v`, `05_register_map_and_driver.md`. *Done when* `bash scripts/run_tiny.sh` and the full flow pass.

**3.5 — Regression.**
`bash scripts/run_dnn_regress.sh` → `ALL DNN CONFIGS PASSED`. Then break something on purpose (drop the rounding term in the PPU, ignore `first_kt`, reverse the weight row order) and **watch the regression fail**: a test suite you have never seen fail is not a test suite. (`bash scripts/run_mutations.sh` does this for 19 prepared faults — read its list for ideas, then invent your own.)

---

## Phase 4 — The software flow (M)

| Step | Do | Learn | Done when |
|---|---|---|---|
| **4.1** Train | `python3 sw/train_mlp.py` | a network is weights + a few functions | test accuracy printed (96.94 %) |
| **4.2** Inspect the ONNX | the 5-line script in `03_software_stack.md` §1, or open in Netron | ONNX = graph + tensors | you can name the 3 nodes and 4 tensors |
| **4.3** Quantize | `python3 sw/quantize.py` | scales, `(mult, shift)`, calibration | float and INT8 accuracy both printed; you can recompute `mult/2^shift` by hand |
| **4.4** Golden model | read `sw/golden.py` (30 lines) next to `rtl/dnn_ppu.v` | bit-exact integer reference | you can point out the RTL line for each Python line |
| **4.5** Compile | `python3 sw/compile.py` | tiling, padding, layout, descriptors | layer table printed; you can explain `src/dst/wbase` |
| **4.6** Simulate + report | `bash scripts/run_all.sh` | the closing of the loop | RTL accuracy = golden accuracy = 96.94 % |
| **4.7** Change the network | `HIDDEN=32 HW_ARGS="--acc-depth 64 --wgt-depth 1024" HW_G="-GACC_DEPTH=64 -GWGT_DEPTH=1024" bash scripts/run_all.sh` — then `HIDDEN=32,16` for three layers | resource limits, ping-pong with 3 layers | bit-exact again (both were run: 96.67 % and 95.83 % INT8 accuracy) |

Read the compile errors! If you choose `HIDDEN=64` (`HIDDEN` is an environment variable of `train_mlp.py`) the compiler stops at the first violated limit — here `layer 0: NT > NT_MAX` — and after you raise that one, the next limit (`M*NT > ACC_DEPTH`, weights `> WGT_DEPTH`, …) is reported. Deciding which hardware parameter to grow, and giving the same value to the RTL parameters, is real architecture work.

---

## Phase 5 — Host: C driver and AXI4-Lite (M)

| Step | Do | Done when |
|---|---|---|
| **5.1** Read the driver | `driver/dnn_accel.h`, `driver/dnn_accel.c` (250 lines) alongside `05_register_map_and_driver.md` | you can say which register write each API call produces |
| **5.2** Prove it | `bash scripts/check_driver.sh` | `IDENTICAL: 568 register accesses` and `DRIVER CHECK PASSED` |
| **5.3** Break it | change one line in the driver (swap two descriptor fields) and rerun | the check fails and shows the first differing line |
| **5.4** AXI4-Lite | read `rtl/axil_reg_bridge.v` (about 120 lines); run `run_all.sh` step 7; try `+axi_stall` | AXI readback identical to direct-bus readback |
| **5.5** Draw the SoC picture | CPU → AXI interconnect → `dnn_accel_axil_top`; where would DMA go? | block diagram in your report |

---

## Phase 6 — ASIC preparation (yours; see `06_verification_and_backend.md`)

1. Decide the silicon configuration (array size, memory depths) and freeze parameters everywhere.
2. Memory macros for weight/activation SRAM; check 1R1W and registered read.
3. SDC from the skeleton, first synthesis at a relaxed clock, then tighten.
4. Find the real critical path (I expect PE MAC, then PPU multiplier).
5. Power: dump activity with `scripts/run_wave.sh`, get power from Genus/Joules/Voltus, compute **pJ per MAC** and **pJ per image**.
6. Formal/coverage additions from chapter 06 §2 if time permits.

---

## Phase 7 — Extensions that add "DNN title impact" (pick one or two)

Each is a real design change with a measurable before/after. Numbers below are **upper bounds derived from the measured cycle split** (FETCH 2, LDW 304, COMP 608, DRAIN 857 of 1 771 cycles per batch of 8).

| # | Extension | What changes | Expected effect | Difficulty |
|---|---|---|---|---|
| **7.1** | **Overlap weight load with compute** — second weight register per PE (shadow), swap with one `weight_swap` pulse | `pe.v`, `systolic_array.v`, `dnn_ctrl.v` | removes LDW's 17 % → ≤ 1.2× | M |
| **7.2** | **Overlap drain with the next pass's LDW/COMP** | `dnn_ctrl.v` (issue next pass before write-backs finish; hazards only if the next pass hits the same accumulator address, i.e. `NT = 1`) | removes most of DRAIN's 48 %; with 7.1, ideal ≈ 2.9× | L |
| **7.3** | **Bigger array / bigger batch** | parameters only; use `python3 sw/sweep_arch.py` | 16×16 cuts bytes/MAC 5×; batch 32 doubles 4×4 utilization (66 % measured) | S |
| **7.4** | **Output-stationary variant** — partial sums stay in the PEs across K | new `pe`, controller | removes the accumulator-SRAM traffic (1.87 of 2.25 B/MAC at 4×4) — the architectural comparison a professor will like | L |
| **7.5** | **Convolution via im2col** | software: unfold the image into a matrix (`sw/`); hardware unchanged | runs a conv layer as a FC layer: the "we support CNNs" claim | M |
| **7.6** | **Per-channel quantization** | `sw/quantize.py`: `mult`/`shift` per output column; PPU takes a per-lane multiplier | accuracy on harder networks | M |
| **7.7** | **Zero-gating** — do not toggle the multiplier when `act == 0` (ReLU outputs are ~40 % zeros; 7 of 16 in the traced example) | `pe.v` (operand isolation) | dynamic power; must be measured with Voltus on a VCD | S–M |
| **7.8** | **INT4 weights** (2 per byte) | packing in software, unpack in `pe` | halves weight storage/bandwidth | M |
| **7.9** | **A bigger dataset** (e.g., MNIST 784→128→10, 101 632 MACs/image) | needs `KT_MAX ≈ 196`, deeper SRAMs; the data set has to be downloaded, which was not possible in the build environment, so this was not run | credible "real workload" | M |
| **7.10** | **Skip layers' host readback / DMA path** | new AXI4 master | throughput for larger models | L |
| **7.11** | **Loop-order study: N-tile outer, K-tile inner** — partial sums stay in one register set, activations are re-read from SRAM for every N-tile (no LRB benefit) | `dnn_ctrl.v` loop nest; software plan unchanged | trades the accumulator-SRAM traffic (1.87 B/MAC at 4×4) against extra activation-SRAM reads; a clean A/B with the same sweep script | M |

For each, the loop is: implement → `run_dnn_regress.sh` stays green → `sweep_arch.py` / perf counters show the change → report.

---

## 8. Fast track: what to show your professor

1. **The claim:** *"A weight-stationary INT8 systolic accelerator that runs a complete trained network on-chip; bit-exact against an integer reference; 96.94 % digit accuracy in RTL simulation."*
2. **The evidence you can produce in an afternoon:**
   * `bash scripts/run_all.sh` output (accuracy chain: float = INT8 golden = RTL).
   * `python3 sw/trace_one.py 0 5` — one image, every number, so you can *explain* the network.
   * A waveform screenshot of `state`, `kt_cnt`/`nt_cnt`, `res_vld` from `run_wave.sh`.
   * The sweep table from `sweep_arch.py` and the "bytes per MAC" discussion.
   * Regression summary: 240 random networks + negative tests + 19 scripted mutation tests (`run_mutations.sh`).
3. **The honest limitations slide:** fully-connected only; no power/area/timing yet; utilization 33 % (with the 7.1/7.2 plan); partial-sum traffic dominates at 4×4 (with the 7.3/7.4 plan).
4. **The numbers only you can add:** area, fmax, power and pJ/inference from Genus/Innovus/Voltus.
