# Systolic-Array INT8 DNN Accelerator — Design Notes

Project: *ASIC Implementation of a Systolic Array Based DNN Accelerator* (guide: Dr. Ravi S)
Deliverable: synthesizable Verilog RTL, testbenches, regression script, and this document (Phase 1 analysis, Phase 3 handoff).

> **Document map.** This file documents the **v1 matrix-multiply core** (first delivery): report analysis, code evaluation, micro-architecture of the systolic core, and Phase 3 handoff notes. It is still accurate for `pe`, `skew_delay`, `systolic_array`, `sa_core`, `lrb`, `accel_ctrl` and `systolic_accel_top`.
> The **DNN engine (v2)** — accumulation across K-tiles, requantization/ReLU, multi-layer sequencing, register interface, AXI4-Lite, software flow, C driver — is documented in `00_START_HERE.md` … `06_verification_and_backend.md`. Note: `sa_core.v` gained one output (`res_vld_early`) in v2; v1 ignores it.

---

## 1. What the report asks for (requirements I extracted)

| # | Requirement (source: report) | Where it lives in the RTL |
|---|---|---|
| R1 | Parameterized systolic array in Verilog | `systolic_array.v` (`ROWS`, `COLS`) |
| R2 | INT8 multiply-accumulate PEs, wide accumulator | `pe.v` (`DATA_WIDTH=8`, `ACC_WIDTH=32`) |
| R3 | **Weight-stationary** dataflow | weight register in each PE, shift-loaded down each column |
| R4 | Local Reuse Buffer (LRB) to cut SRAM accesses | `lrb.v` |
| R5 | Hierarchical on-chip memory | 3 SRAMs (`sram_sdp.v`) + LRB + PE weight registers |
| R6 | Verify against a software reference | `tb_systolic_accel.v` (in-TB golden model), `tb_pe.v` |
| R7 | Synthesis / STA / area-power-throughput-latency metrics | RTL is Genus-ready; `perf_counters.v` reports cycles and memory reads |

**Reading of the report, and assumptions I had to make** (tell me if any is wrong):

* INT8 means **signed two's-complement**. The report never says so, but signed is the norm for quantized inference, and the report's own reference [4] (Jacob et al.) uses it.
* The report's images show only the PE and its testbench. The array, memories, LRB and control were described in prose, so I designed them.
* The abstract says the LRB captures "reusable weight **and** activation values". In a weight-stationary array the weights are already reused *M* times inside the PE registers, so the LRB that actually removes memory traffic is the **activation-side** one. I built that one and did not add a redundant weight LRB.
* The report says "Local Reuse Buffers" (plural) and a "hierarchical memory". I implemented one LRB and a three-level hierarchy: SRAM → LRB → PE registers.
* No OCR corrections were needed. The code images were clean and I found no syntax errors in them. The changes I made are functional and synthesis-related (next section).

---

## 2. Code evaluation: keep / modify / rewrite

| Item from report | Verdict | Why |
|---|---|---|
| **PE dataflow** (`act_out <= act_in`, `psum_out <= psum_in + w*act`, `weight_load`) | **Kept** | This is exactly the right weight-stationary PE. Both outputs are registered, so the hop timing is one cycle per PE, which is what the skew logic relies on. |
| **Parameters, port names** (`DATA_WIDTH`, `ACC_WIDTH`, `act_in`, `psum_out`, …) | **Kept** | Same names, so your existing waveforms and report text still map onto the RTL. |
| **Unsigned multiply** (`weight_reg * act_in` on unsigned regs) | **Rewritten (functional bug)** | For INT8 this is wrong for any negative operand. I ran your PE with weight = −3, act = 7: it returns **1771**, the correct answer is **−21**. Your report's tests all used positive numbers, so the bug never showed. Now `signed` ports/regs and an explicit sign-extended product. |
| **Async reset** (`posedge clk or negedge rst_n`) | **Modified** | Now synchronous, as you asked. The one documented exception is `rst_sync.v` (async-assert / sync-release) at the top level. |
| **No enable** | **Added `en`** | Lets the whole array idle. Genus can turn the enable pattern into ICG clock gating (`lp_insert_clock_gating`). |
| **No path to load weights in an array** | **Added `weight_out`** | The weight register doubles as a shift-chain stage, so loading a column costs no extra flops and no address decoders. |
| **`pe_tb.v`** | **Rewritten** as `tb_pe.v` | The original only printed values; it never checked anything. Kept your 4 vectors (5×3+0, 5×4+10, 5×7+20), added signed corners (−128×−128, −128×127), weight-hold, enable-freeze and reset checks. Self-checking, `PASSED/FAILED`. |
| Array, skew, controller, LRB, SRAMs, top | **New** | Not in the report's images. |

---

## 3. Micro-architecture

### 3.1 Module hierarchy

```
systolic_accel_top                 top level, single clock domain
 ├─ rst_sync                       async-assert / sync-release reset (only async element)
 ├─ accel_ctrl                     FSM: IDLE → LDW → COMP → DRAIN (per weight tile)
 ├─ sram_sdp  u_wgt_mem            weight buffer     COLS×INT8 per word
 ├─ sram_sdp  u_act_mem            activation buffer ROWS×INT8 per word
 ├─ sram_sdp  u_out_mem            result buffer     COLS×INT32 per word
 ├─ lrb                            Local Reuse Buffer + source mux + feed register
 │   └─ sram_sdp                   LRB storage (LRB_DEPTH vectors)
 ├─ sa_core                        compute core
 │   ├─ skew_delay (DESKEW=0)      input skew:  row r delayed r cycles
 │   ├─ systolic_array             ROWS×COLS mesh
 │   │   └─ pe × (ROWS·COLS)       INT8 MAC processing element
 │   ├─ skew_delay (DESKEW=1)      output deskew: column c delayed COLS−1−c
 │   └─ valid pipeline             (ROWS+COLS−1) deep
 └─ perf_counters                  optional (PERF_EN)
```

### 3.2 What the hardware computes

Each pass computes `C = A × W` for one weight tile:

* `A` is a stream of **M activation vectors**, each of length `K = ROWS`.
* `W` is one `ROWS × COLS` tile held stationary in the PEs.
* Each output vector has `COLS` INT32 values.
* Tiles repeat `T` times to cover `N = T·COLS` output columns.

Full problem: `C[M × N] = A[M × ROWS] × W[ROWS × N]`, INT8 in, INT32 out.

> **Limitation to be aware of:** the reduction depth `K` is fixed at `ROWS`. There is no accumulation *across* K-tiles. A real layer with K > ROWS needs an accumulate-into-output-SRAM path. It is listed on the roadmap in section 11.

### 3.3 Data path, in plain terms

Think of an assembly line.

* Each PE is a **worker at a fixed station** holding one number (the weight).
* Activations are **items travelling left-to-right on a conveyor belt**.
* A **running total travels top-to-bottom on a second belt**.
* At every station: `new_total = total_from_above + (my_weight × item_passing_by)`.

For the totals to be right, item *r* has to reach row *r* at exactly the moment the running total from row *r−1* arrives. That is why the inputs are **staggered (skewed)**: row 0 starts immediately, row 1 one cycle later, row 2 two cycles later, and so on. The mirror-image stagger at the output (**deskew**) lines the columns back up.

The **LRB** is a small notepad next to the line. If you'll need the same items again for the next batch of stations (the next weight tile), you copy them down once and read them from the notepad instead of walking back to the warehouse (the big SRAM).

### 3.4 Traced example (real RTL output)

2×2 array, `A = [[1,2],[3,4]]` (M=2, K=2), `W = [[5,6],[7,8]]`.
Expected `C = [[1·5+2·7, 1·6+2·8],[3·5+4·7, 3·6+4·8]] = [[19,22],[43,50]]`.

Steps in the trace below (from `tb/trace_2x2.v`, cycle numbers are real):

| Cycle | What happens |
|---|---|
| 1–2 | `wgt_load=1`. Weight row 1 (`[7,8]`) is read **first** and shifts down; row 0 (`[5,6]`) follows. |
| 3 | Weights settled: `W=[[5,6],[7,8]]`. |
| 4 | Vector `[1,2]` enters the core. Row 0 gets `1` now, row 1's `2` is held one cycle. |
| 5 | Vector `[3,4]` enters. Skewed rows are `[3,2]`: row 1 is now seeing the *previous* vector's `2`. |
| 6 | Bottom psum col 0 = **19** (= 1·5 + 2·7). |
| 7 | Col 0 = **43**, col 1 = **22**. Deskew has delayed col 0's 19 by one cycle, so `res_vld=1` with `[19,22]` together. |
| 8 | `res = [43,50]`. Done. |

```
cyc | st wl | actvld vec   | skewed rows | PE weights        | bottom psums | res_vld res
  1 |  1  1 |   0    [0,0]  |   [0,0]     | [[0,0],[0,0]]     |   [0,0]      |   0    [0,0]
  2 |  2  1 |   0    [0,0]  |   [0,0]     | [[7,8],[0,0]]     |   [0,0]      |   0    [0,0]
  3 |  2  0 |   0    [0,0]  |   [0,0]     | [[5,6],[7,8]]     |   [0,0]      |   0    [0,0]
  4 |  3  0 |   1    [1,2]  |   [1,0]     | [[5,6],[7,8]]     |   [0,0]      |   0    [0,0]
  5 |  3  0 |   1    [3,4]  |   [3,2]     | [[5,6],[7,8]]     |   [0,0]      |   0    [0,0]
  6 |  3  0 |   0    [0,0]  |   [0,4]     | [[5,6],[7,8]]     |   [19,0]     |   0    [0,0]
  7 |  3  0 |   0    [0,0]  |   [0,0]     | [[5,6],[7,8]]     |   [43,22]    |   1    [19,22]
  8 |  3  0 |   0    [0,0]  |   [0,0]     | [[5,6],[7,8]]     |   [0,50]     |   1    [43,50]
```

(`st`: 1=LDW, 2=COMP, 3=DRAIN.)

**Timing rule (verified by the exact cycle-count check in the testbench).** A vector entering the core at cycle τ produces its result at τ + (ROWS + COLS − 1), all columns aligned.

### 3.5 Traced example at the default 4×4 size, with shapes

Problem: `A[8×4] int8 × W[4×16] int8 → C[8×16] int32`, so `M=8`, `T=16/4=4`.

| Item | Shape / width | Value |
|---|---|---|
| Activation SRAM word | `ROWS × INT8` | 32 bits, 8 words used |
| Weight SRAM word | `COLS × INT8` | 32 bits, 16 words used (4 tiles × 4 rows) |
| Output SRAM word | `COLS × INT32` | 128 bits, 32 words used (4 tiles × 8 rows) |
| LRB | 8 vectors × 32 bits | all 8 activation vectors fit |
| Cycles per tile | 4 (LDW) + 8 (COMP) + 10 (DRAIN) | **22** → 4 tiles = **88 cycles** (matches test 9 exactly) |
| Activation-SRAM reads | tile 0 only | **8** (vs 32 without an LRB, a 4× reduction) |
| LRB reads | tiles 1–3 × 8 | 24 |
| Weight-SRAM reads | 4 tiles × 4 rows | 16 |

General formulas (checked by the testbench on every run):

```
cycles         = T · (M + 2·ROWS + COLS + 2)
act SRAM reads = M + (T−1)·max(0, M − LRB_DEPTH)
LRB reads      = (T−1)·min(M, LRB_DEPTH)
weight reads   = T · ROWS
peak throughput= ROWS·COLS MAC/cycle          (16 MAC/cycle at 4×4)
utilization    = M / (M + 2·ROWS + COLS + 2)  (53 % at 4×4, M=16)
```

---

## 4. Finite-state machine (`accel_ctrl`)

```
              start & cfg_ok
   IDLE ─────────────────────▶ LDW ───────────▶ COMP ───────────▶ DRAIN
    ▲                           ▲   ROWS cycles    M cycles           │
    │                           │                                     │
    │                           └──── more tiles (all M results in) ──┤
    └──────────── last tile: done=1 ──────────────────────────────────┘
```

| State | What it does | Exit condition |
|---|---|---|
| **IDLE** | Waits for `start`. Validates config combinationally. Bad config → `cfg_err=1`, stay idle. Good config → latch M/T, clear pointers, `done=0`. | `start & cfg_ok` |
| **LDW** | Reads `ROWS` weight-SRAM words, **row ROWS−1 first**. The registered read-enable (`wgt_load`) shifts them down the columns; the first word ends at the bottom row. | `ld_cnt == ROWS−1` |
| **COMP** | Issues `M` activation reads. Tile 0 reads the SRAM and fills the LRB (vectors with index < `LRB_DEPTH`). Later tiles read the LRB for index < `LRB_DEPTH`, SRAM otherwise. | `m_cnt == M−1` |
| **DRAIN** | Waits until all `M` results of this tile have been written to the output SRAM (`res_cnt == M`). Zeros flush the mesh meanwhile, so the next weight load starts on a clean pipeline. | `res_cnt == M`; last tile → IDLE (`done=1`), else → LDW |

Config is valid when `1 ≤ M ≤ ACT_DEPTH`, `1 ≤ T ≤ WGT_DEPTH/ROWS`, and `M·T ≤ OUT_DEPTH`.

Memory layouts (all read latency = 1 cycle):

| Memory | Address | Word |
|---|---|---|
| Weight SRAM | `tile·ROWS + r` | `W[r][c]` at bits `[c·8 +: 8]` |
| Activation SRAM | `m` | `A[m][r]` at bits `[r·8 +: 8]` |
| Output SRAM | `tile·M + m` (running pointer, no multiplier) | `C[m][tile·COLS + c]` at bits `[c·32 +: 32]` |

Host rules: `start` must be a **single-cycle pulse** (a held `start` re-launches when the run finishes); memory writes are ignored while `busy`; `done` is sticky until the next `start`.

## 5. Key internal interfaces and timing

| Interface | Signals | Timing note |
|---|---|---|
| ctrl → weight SRAM | `wgt_rd_en`, `wgt_rd_addr` | data valid next cycle |
| ctrl → PEs | `wgt_load` | = `wgt_rd_en` delayed 1 cycle, so it lines up with the SRAM read data |
| ctrl → act SRAM / LRB | `act_rd_*`, `lrb_rd_*` | one of them fires per vector (never both) |
| ctrl → LRB (side-band) | `lrb_wr_en/addr`, `feed_vld`, `feed_sel_lrb` | delayed 1 cycle so they align with read data (`feed_*_r` regs) |
| LRB → core | `act_vld`, `act_vec` | registered mux output: read issued at c0, vector at the core input at **c0+2** |
| core → ctrl / out SRAM | `res_vld`, `res_vec` | vector entering at τ appears at **τ + ROWS + COLS − 1**; no back-pressure, consumed same cycle |
| host → memories | `host_*_we/addr/data` | synchronous to `clk`, blocked while `busy` |

**Reset / CDC.** Single clock domain, so there is no CDC inside the design. The only asynchronous signal is `rst_n_i`, handled by `rst_sync` (async assert, sync release). All host inputs must be synchronous to `clk`. The clock must run during reset, because every downstream reset is synchronous.

## 6. Design decisions worth knowing

1. **Shift-load weights** instead of per-PE write enables. There are no address decoders and no long enable wires, but a tile load costs `ROWS` cycles and can't overlap compute (single weight buffer).
2. **Global `en` = `busy`.** Everything freezes together when idle. Consistent pipeline state, and one clean enable for clock gating.
3. **Mux + register in front of the array** (in `lrb`). It adds a cycle of latency but keeps the SRAM/LRB mux off the multiplier's critical path.
4. **Partial LRB caching.** If `M > LRB_DEPTH`, the first `LRB_DEPTH` vectors are replayed from the LRB and the rest still come from SRAM. There is no cliff at the boundary, and the read counts follow the formula in section 3.5.
5. **Behavioral SRAMs** (`sram_sdp`). Genus maps them to flip-flops. That is fine at the default size (about 5.4 K bits) but you should swap in foundry macros before scaling up (keep the port list).
6. **Sync reset on datapath flops** (PEs, skew registers) keeps simulation X-free and behavior deterministic. If you later want to trim area, the datapath regs are the place: the mesh flushes itself with zeros within a few cycles.

## 7. Deliverables and verification status

```
systolic_dnn_accel/
├─ rtl/   pe.v  skew_delay.v  systolic_array.v  sa_core.v  sram_sdp.v  lrb.v
│         accel_ctrl.v  rst_sync.v  perf_counters.v  systolic_accel_top.v
├─ tb/    tb_systolic_accel.v  tb_pe.v  trace_2x2.v
├─ scripts/run_regress.sh
└─ docs/  DESIGN_NOTES.md
```

Top-level parameters (`ROWS, COLS, DATA_WIDTH, ACC_WIDTH, ACT_DEPTH, WGT_DEPTH, OUT_DEPTH, LRB_DEPTH, PERF_EN, SYNC_RST_IN`). The `*_W` / `*_AW` parameters are derived, do not override.

**What I actually ran** (results, not a promise):

| Check | Result |
|---|---|
| `iverilog -g2005 -Wall` compile | clean |
| `verilator --lint-only -Wall` | clean (a few intentional waivers marked inline) |
| `tb_pe` (signed corners, hold, enable, reset) | PASSED |
| `tb_systolic_accel`, about 80 tests per config: directed data corners (−128, +127, zero, mixed), shape corners, full/partial/no LRB hit, invalid configs, start-while-busy, writes-while-busy, 60 random back-to-back runs with no reset, async reset mid-run | **PASSED on 9 configurations** (4×4, 8×8, 3×5, 6×2, 2×8, 1×1, 16×16, LRB_DEPTH = 1 / 8 / 32) |
| Exact schedule checks on every run | cycle count, weight/act-SRAM reads and LRB reads all match the formulas |
| Simulators | Verilator (all configs) and Icarus (seven small configs), same result |
| `yosys check -assert` + generic `synth -flatten` | 0 latches, no loops or multi-driver nets; ≈6.6 K flops, of which 5,376 are SRAM storage bits and only **2** use async reset (`rst_sync`) |

Not done: I have **not** run Genus, Innovus or Tempus, so I have no library timing, area or power numbers. The Yosys run is only a structural sanity check against a generic gate set.

`$clog2` appears in the top-level parameter defaults (Verilog-2005). If your Genus `read_hdl` mode rejects it, read with `-sv` or hard-code the six derived parameters.

---

## 8. Phase 3a — Verification suggestions (before you synthesize)

The testbench already covers the items in section 7. These are the **next** things worth adding, roughly in priority order.

| # | Edge case / test | Why it matters |
|---|---|---|
| 1 | **Identity weights** (`W = I`, so `C = A`) and **one-hot activations** | My uniform-data corners (all −128, all +127) can't reveal a swapped row or column. Random data does, but identity/one-hot makes a routing bug obvious at a glance. |
| 2 | **Reset in every FSM state** (LDW, COMP, DRAIN) and on the exact cycle `res_vld` is high | The current TB resets once mid-run. A randomized reset time sweeps every state. |
| 3 | **Boundary sweeps:** `M ∈ {1, LRB_DEPTH−1, LRB_DEPTH, LRB_DEPTH+1, ACT_DEPTH}`, `T ∈ {1, MAX}`, `M·T == OUT_DEPTH` and `OUT_DEPTH+1` (must give `cfg_err`) | Off-by-one in the LRB partial-hit policy and the output pointer. |
| 4 | **Handshake corners:** `start` on the cycle right after `done`; `start` held high for several cycles (re-launches, by design); a host write in the *same* cycle as `start` | Documents the exact host contract. |
| 5 | **Host reads output SRAM while busy** | Allowed by design. Confirm no corruption. |
| 6 | **Minimum accumulator width:** set `ACC_WIDTH = 2·DATA_WIDTH + clog2(ROWS)` (18 bits at 4 rows, 20 at 16) and rerun the −128×−128 corner | Proves the smaller adder is safe if you trim area (section 9). |
| 7 | **X-propagation:** gate-level or X-init run where SRAM contents start as X | The design must never read an unwritten location if the host loads `M` activation words and `T·ROWS` weight words. Confirms no X reaches `host_out_data`. |
| 8 | **Independent golden model:** NumPy `int8 @ int8 → int32` writing vector files, read with `$readmemh` | The report plans a NumPy reference. My in-TB golden is a plain triple loop; a second implementation removes common-mode risk. |
| 9 | **A real quantized layer:** conv/FC layer → im2col → tiles → compare against NumPy | Shows the application-level result the report's motivation asks for. |
| 10 | **Post-synthesis / post-PnR gate-level sim with SDF** using `tb_systolic_accel` unchanged | The TB has no hierarchical peeks, so it runs on a netlist. (`trace_2x2.v` does peek and is RTL-only.) Synthesize the specific parameter set you simulate. |
| 11 | **Coverage:** functional (shape × LRB regime × data class × cfg_err reason), plus FSM/toggle coverage | Toggle data doubles as power-analysis input. |

**Assertions worth writing** (SVA, usable in simulation and in JasperGold):

1. `start && !busy && cfg_ok |=> busy`
2. Bounded liveness: `busy` falls within `T·(M+2·ROWS+COLS+2)+k` cycles
3. Exactly `M·T` `res_vld` pulses per run
4. `out_wr_en |-> out_wr_addr < OUT_DEPTH`
5. `!(act_rd_en && lrb_rd_en)`
6. An LRB read only hits a slot filled earlier in the same run
7. `busy |-> !(effective host write enable)`
8. `cfg_err |-> !busy`
9. First `res_vld` is exactly `ROWS+COLS−1` cycles after the first `feed_vld` (+1 for the feed register)
10. FSM state is always one of the four legal encodings

---

## 9. Phase 3b — Constraint considerations (SDC concepts for this architecture)

| Topic | What to keep in mind |
|---|---|
| **Clock** | One clock. The critical path is almost certainly the **PE MAC**: signed 8×8 multiply plus a 32-bit add in one cycle, reg→reg. Everything else (skew chains, controller compares, mux stage) is shallow. Confirm in the first timing report. |
| **Accumulator width** | For `K = ROWS` the accumulator needs only `2·DATA_WIDTH + clog2(ROWS)` bits (18 at 4 rows). 32 is 14 bits wider than needed *today*, in every PE. Sweeping `ACC_WIDTH` shortens the carry chain and cuts area and power. Keep 32 only if you add K-tile accumulation. |
| **Multiply-add merging** | `psum_in + w·act` is a fused multiply-add. Check that the datapath synthesis merged the add into the multiplier's compressor tree (carry-save) rather than building multiplier then adder. |
| **Reset (`rst_n_i`)** | Asynchronous port. Treat it as a false path into the two `rst_sync` flops' async pins (a recovery/removal check against an unrelated external source is meaningless) and mark those two flops don't-touch. The synchronized `rst_n` then fans out to thousands of flops (PEs, skew chains, SRAM read registers), so plan a **buffered reset tree**, or register a per-block copy. |
| **Clock gating** | Every PE, skew and feed flop shares `en = busy`, so this is a natural ICG target (`set_db lp_insert_clock_gating true` in Genus). **Check that the inferred gate enable includes reset** (`en \| ~rst_n`). These flops use synchronous reset, and `busy` is low during reset, so a gate driven by `en` alone would stop the clock and the flops would never reset. `busy` is a decode of the 2-bit state register, so it arrives early enough for the ICG setup check. |
| **High-fanout nets** | `busy` and `wgt_load` reach every PE (`wgt_load` → `ROWS·COLS·8` flops). Expect buffer trees. Set sensible `max_fanout`/`max_transition`, and let CTS handle the gated branches. |
| **I/O timing** | All host inputs are synchronous to `clk`: use `set_input_delay` / `set_output_delay` against it. `host_out_data` leaves straight from SRAM output flops (good). `busy` is a small state decode; `done`/`cfg_err` are flops. |
| **The one real input→logic path** | `cfg_m × cfg_tiles → compare → state / cfg_err`, straight from ports. Shallow but not free. Cleanest fix: register `cfg_m`/`cfg_tiles` first (a one-line RTL change). A multicycle exception is valid **only if** the host holds them stable ≥2 cycles before `start`. The RTL doesn't enforce that. |
| **Pin count** | Default 4×4 has about **350 signal pins**: `host_out_data` 128, perf counters 128, weight/activation write buses 32 each plus addresses. Fine for a block-level flow. For a pad ring, set `PERF_EN=0` (or mux the counters through a small register interface) and serialize the host memory ports. |
| **Memory macros** | When you replace `sram_sdp` bodies with macros: add their `.lib/.lef`, set don't-touch, and check the read latency. The controller assumes **1-cycle registered read**. A 2-cycle macro needs one more delay stage on the `feed_*` / `wgt_load` side-bands. |
| **Hierarchy / floorplan** | The PE mesh is a regular grid. At 4×4, flatten freely. At 8×8 and up, consider keeping `pe` (or `systolic_array`) as a hierarchy boundary so placement and repeated-cell optimization stay predictable. |
| **Hold** | Skew and deskew are pure register chains (reg→reg with no logic), and PE→PE act/weight hops are the same. Expect **hold-fix buffers** after CTS. That is normal, not an RTL defect. |
| **DFT** | Scan insertion needs test-mode control of reset and the ICG enables; give the ICGs a test-enable. The 5.4 K SRAM flops inflate the scan chain, another reason to move to macros with MBIST. |
| **False / multicycle paths** | None needed structurally. `m_lat`/`t_lat` are quasi-static during a run, but I would not declare an exception on them. |

---

## 10. Phase 3c — Tool and flow enhancements

**Formal and equivalence (Cadence-centric)**

* **JasperGold FPV** on `accel_ctrl` + LRB: prove the assertions in section 8 (deadlock-freedom, exactly `M·T` results, no out-of-range write, LRB read-after-fill). The controller is small, so full proofs are realistic.
* **JasperGold SEC** (sequential equivalence): when you later refactor (pipelined MAC, double-buffered weights, narrower `ACC_WIDTH`), prove the new RTL equivalent to this baseline under the same config. That protects you from silent regressions.
* **Conformal LEC** at each hand-off: RTL vs Genus netlist, then netlist vs post-PnR (including ECOs).
* **JasperGold X-prop** or an X-init gate-level run for the reset/SRAM-uninitialised question (item 7 above).
* If you ever add a second clock (e.g. a slower host interface), run a **CDC** analysis. Today the design has none.

**Simulation, coverage, power**

* **Xcelium + IMC/vManager** for coverage-driven regressions; reuse `run_regress.sh`'s configuration matrix as the regression list.
* **Joules** (RTL/gate power) with SAIF/FSDB from three workloads: all-zero, dense random, sparse. It shows data-dependent power and the value of clock gating; **Voltus** for IR/EM once placed.
* **Post-layout GLS with SDF** from Innovus/Tempus using the unchanged TB.

**Methodology that will pay off for this report specifically**

* **Design-space sweep:** script Genus batch runs over `ROWS×COLS ∈ {4×4, 8×8, 16×16}`, `ACC_WIDTH ∈ {18/20, 32}`, `LRB_DEPTH`, `PERF_EN`, and tabulate area / power / fmax / throughput. That table *is* your "evaluate area, power, throughput and latency" objective, and it makes the trade-offs (array size vs. LRB vs. accumulator width) explicit.
* **NumPy / PyTorch golden flow:** generate quantized layer vectors and expected outputs in Python, run them through simulation via `$readmemh` (or cocotb). This is the reference flow the report already names.
* **CI:** Verilator lint plus `scripts/run_regress.sh` on every commit (Verilator: about 20 s to build, then well under a second per config; Icarus is slow for arrays above ~6×6 because it re-evaluates wide flattened buses).
* **Optional free cross-check:** OpenROAD-flow-scripts with a public PDK for an early look at congestion and timing while institute licences are queued.

---

## 11. Known limitations and roadmap (ranked by value)

1. **Overlap weight load and drain with compute** (double-buffered weights, back-to-back tile scheduling). Today each tile pays `2·ROWS + COLS + 2` idle-ish cycles, so utilization is `M/(M+2·ROWS+COLS+2)` (53 % at 4×4, M=16). Overlapping lets it approach the peak for streams of tiles. This is the biggest throughput win.
2. **K-tile accumulation** (`C += partial`). Today `K = ROWS` only; real layers with a deeper reduction need an accumulate-into-output-SRAM path (and then the 32-bit accumulator is justified).
3. **Output post-processing:** bias, ReLU, saturating requantization INT32 → INT8. It cuts result bandwidth 4× and is what the next layer consumes.
4. **Narrow host interface** (AXI-Lite for control plus a stream/DMA port) to remove ~250 pins.
5. **Pipelined MAC for higher fmax.** Splitting multiply and add makes each vertical hop 2 cycles, so the input skew becomes `2·r` and latency grows to `2·ROWS+COLS−1`. That is a contained change to `skew_delay` parameters and the valid-pipeline depth.
6. **Zero-skipping / operand isolation** per PE for power on sparse activations.
7. **Foundry SRAM macros + MBIST**, replacing the behavioral memories.

---

*Provenance:* every number in sections 3.4, 3.5 and 7 comes from a simulation or tool run in this project (`tb/trace_2x2.v`, `tb/tb_systolic_accel.v`, `scripts/run_regress.sh`, Yosys). Nothing here comes from a Cadence tool run.
