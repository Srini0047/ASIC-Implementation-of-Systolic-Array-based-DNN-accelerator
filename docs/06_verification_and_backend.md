# 06 — Verification Report and Back-End Handoff (v2 supplement)

*`DESIGN_NOTES.md` (Phase 1 & 3 notes) covers the v1 matrix-multiply core. This chapter is its counterpart for the DNN engine: what was verified and how strongly, what was **not** verified, and what is specific to this design when you take it through Genus / Innovus / Tempus. You said you drive the back end yourself, so this is a checklist of design-specific facts, not a tutorial.*

**Honesty note.** No Cadence tool was available in the environment where this was built. Everything below was run with Verilator 5.020, Icarus Verilog 12, Yosys 0.33, GCC 13 and Python 3.11. There are **no synthesis, timing, area or power numbers** in this delivery; where I mention likely critical paths or memory sizes those are reasoned from the RTL, not measured.

---

## 1. What was verified

| # | Test | Command | What it proves | Result |
|---|---|---|---|---|
| 1 | PE unit test | `tb/tb_pe.v` | signed MAC corners (−128 × −128, −128 × 127), weight hold, enable freeze, reset | PASS |
| 2 | v1 core regression | `scripts/run_regress.sh` (≈ 6 min) | matrix-multiply core on 9 configurations (1×1 … 16×16, non-square, LRB depth 1 … 32), 79–80 tests each (60 randomized + directed corner cases), cycle counts checked against the closed form; Verilator on all 9, Icarus additionally on the 7 with ROWS·COLS ≤ 24 | ALL PASSED |
| 3 | Digit network, whole flow | `scripts/run_all.sh` | PyTorch-trained network → INT8 → register program → RTL: 360 images. Of the 1 313 reads, 1 088 are compared word-for-word with the integer golden (1 080 output words + 8 ID/configuration reads); the other 225 are performance-counter reads that are logged and used for the report, not compared | 0 mismatches; **96.94 %** float = INT8 golden = RTL |
| 4 | Random multi-layer regression | `scripts/run_dnn_regress.sh` (≈ 1 min) | 240 random networks (1–4 layers; random K, N, batch; extreme weights; random bias, `mult`, `shift`, ReLU) on 7 hardware configurations (ROWS = 1, 2, 3, 4, 4, 4, 8; varying `M_MAX`, `KT_MAX`, `NT_MAX`, LRB depth incl. 1) **plus** 116 negative tests (fixed seeds, so the count is reproducible): each is an illegal command — a descriptor with `M`, `KT` or `NT` of 0 or max+1, `M·NT > ACC_DEPTH`, `SHIFT` = 48, or an `N_LAYERS` of 0, `NUM_LAYERS+1` or a value that would wrap to a legal count if truncated (the layer-count tests run while the descriptors are legal, so only a rejected count can raise `err`) — each must raise `err` (and STATUS must read exactly `0x9` — busy = 1, done = 0, err = 0, rd_valid = 1 — on the first read after every START, i.e. no stale `err`/`done`) | ALL PASSED |
| 5 | C driver ≡ Python compiler | `scripts/check_driver.sh` | the C driver's 568-access register stream is byte-identical to the Python one, and replays on RTL | PASS |
| 6 | AXI4-Lite bridge | `run_all.sh` step 7, `run_dnn_regress.sh` last line | whole network + random program through AXI4-Lite with independent AW/W channels and random `BREADY`/`RREADY`/gap delays; readback identical to direct bus | PASS |
| 7 | Architecture sweep | `python3 sw/sweep_arch.py` | 5 configs (4×4 … 16×16, batch 8 / 32) all produce correct results (`PLAYER PASSED`) and give the performance table in chapter 2 | PASS |
| 8 | Hand-checkable example | `bash scripts/run_tiny.sh` | `[1 2 3 4]·W + bias` = `[90 100 110 0]` computed by the RTL | PASS |
| 9 | Lint | `scripts/check_rtl_quality.sh` (Verilator `--lint-only -Wall` on `dnn_accel_top` and on `dnn_accel_axil_top`, all `rtl/` files of that top) | 0 warnings on both tops (a few documented width/unused-signal waivers in file headers) | clean |
| 10 | Structure | `scripts/check_rtl_quality.sh` (Yosys: `read_verilog …; hierarchy -check; proc; flatten; check -assert` + a count of latch and async-reset cells) | no latches inferred; exactly one asynchronous-reset cell (`rst_sync`) | clean |
| 11 | Partial-sum formula | `scripts/check_psum_formula.sh` and `python3 sw/sweep_arch.py` | the traffic formula behind the bytes-per-MAC numbers equals the RTL's real `acc_wr_en` / `acc_rd_en` strobe counts (24 840 = 24 840 for the default run; all 5 sweep configurations) | PASS |

### Are the tests strong enough? Mutation testing

A passing test suite only means something if it fails on a broken design. `bash scripts/run_mutations.sh` (≈ 2–3 min) injects one fault at a time into a *copy* of the design and requires the tests to go red. Nothing in `rtl/` or `driver/` is modified; every mutation is an exact text replacement listed in the script, and the script fails if a pattern no longer matches the source. All 19 were caught when this delivery was finalized:

| Group | Injected fault | Detected by |
|---|---|---|
| RTL datapath / control | `dnn_ppu.v` rounding term removed · bias not added · `dnn_accel_core.v` stale partial sum not masked on the first K-tile · `dnn_ctrl.v` activation address stride off by one · weight rows loaded in the wrong order · `sa_core.v` `res_vld_early` one cycle too late · one cycle too early | default digit-network program (bit-exact compare) |
| RTL error handling | stale `err` visible right after START · `N_LAYERS` out-of-range value wraps instead of clamping · `M·NT ≤ ACC_DEPTH` check removed · `SHIFT` range check removed | random program with negative tests |
| C driver | descriptor SRC/DST swapped · weight rows reversed · bias lanes reversed · output read-back stride wrong | driver register stream vs. Python stream |
| AXI4-Lite bridge | `AWREADY` without `WVALID` · `WREADY` without `AWVALID` · wrong write-address slice · `RVALID` one cycle early | default program through AXI with random stalls |

The mutants are deliberately simple; killing them shows the checks are sensitive to these fault classes, not that the design is bug-free.

### What is **not** verified (be explicit about this in your report)

* No gate-level or SDF simulation; no equivalence check RTL ↔ netlist.
* No X-propagation study (Verilator is 2-state). The datapath registers are reset, but memories are not.
* No formal verification, no SVA assertions, no functional-coverage collection, no code-coverage numbers.
* Only fully-connected layers; no convolution.
* Host writes while `busy`: `dnn_accel_core` gates weight/bias writes (`wgt_we & ~busy`, `bias_we & ~busy`) and switches the activation-SRAM write port to the PPU while busy, and `dnn_ctrl` ignores descriptor writes while busy — but no test deliberately hammers these ports mid-run, so the *behaviour* (writes ignored) is by inspection, not by test. Host **reads** of the activation SRAM while busy are not supported (the read port belongs to the controller).
* Not run under Xcelium/Questa/VCS — the testbench uses only Verilog-2001 plus `$urandom` / `$test$plusargs` / `$value$plusargs`, so it should port, but that is untested.

---

## 2. Assertions and formal properties worth adding (JasperGold / SVA)

Concrete, small, and high-value. Bind them to `dnn_ctrl`, `dnn_reg_if` and `axil_reg_bridge`.

**Control FSM (`dnn_ctrl`)**

```systemverilog
// 1. State encoding stays legal
a_state_legal:   assert property (@(posedge clk) disable iff (!rst_n) state inside {IDLE,FETCH,LDW,COMP,DRAIN});
// 2. Progress: every START terminates (done or err) - bounded-liveness in Jasper
a_start_done:    assert property (@(posedge clk) disable iff (!rst_n) start |-> ##[1:MAX_CYC] (done || err));
// 3. done and err are exclusive
a_done_xor_err:  assert property (@(posedge clk) !(done && err));
// 4. Counters never exceed the descriptor
a_kt_range:      assert property (@(posedge clk) state==COMP |-> kt_cnt < kt_l && nt_cnt < nt_l && m_cnt < m_l);
// 5. An accumulator read only ever happens after the first K-tile
a_acc_rd_kt:     assert property (@(posedge clk) acc_rd_en |-> !first_kt);
// 6. Every vector fed is later written back exactly once (wr_done count == m_l at DRAIN exit)
```

**Register interface**  *(signal names are illustrative — `desc`, `status_rd` stand for the descriptor array in `dnn_ctrl` and the value returned for the STATUS register in `dnn_reg_if`; adapt them to the RTL)*

```systemverilog
// dnn_reg_if raises desc_we whether or not the core is busy; it is dnn_ctrl that ignores it.
// So the property belongs on the control side: a descriptor write while busy must not change the descriptor.
a_desc_frozen:   assert property (@(posedge clk) (busy && desc_we) |=> $stable(desc));      // bind to dnn_ctrl
// START is not guaranteed to be a single-cycle pulse (two back-to-back CTRL writes keep it high two cycles);
// what must hold is that the core never restarts while already busy:
a_no_restart:    assert property (@(posedge clk) disable iff (!rst_n) (busy && start) |=> busy);
// After START, STATUS must never show a completed/stale result:  busy is visible and done/err are masked
a_status_race:   assert property (@(posedge clk) start_r |-> (status_rd[0] && !status_rd[1] && !status_rd[2]));
```

**AXI4-Lite bridge** — run Jasper's AXI protocol checkers (or write four properties): `AWVALID`/`WVALID`/`ARVALID` stay high until the matching `READY`; `RDATA` and `BVALID`/`RVALID` stay stable until accepted; every accepted request eventually gets a response.

**Formal apps to run:** connectivity (register map → core ports), X-propagation on outputs after reset, sequential equivalence RTL-vs-RTL if you retime/pipeline the MAC, and `dnn_ppu` vs a 20-line C model via C-to-RTL equivalence if you want a mathematical proof of the arithmetic.

**Coverage to define:** every FSM state and transition; KT ∈ {1, 2, 3, KT_MAX}; NT likewise; `M` = 1, `LRB_DEPTH`, `LRB_DEPTH + 1`, `M_MAX`; `SHIFT` = 0, 1, 47; saturation high and low; ReLU on/off; each rejection reason of the descriptor validator; `err` followed by a legal run; back-to-back runs.

---

## 3. Design facts that matter for synthesis (all designs, all tools)

| Topic | Fact |
|---|---|
| Top module | `dnn_accel_top` (register bus) or `dnn_accel_axil_top` (AXI4-Lite SoC wrapper) |
| RTL files | `pe skew_delay systolic_array sa_core sram_sdp lrb perf_counters rst_sync dnn_ppu dnn_ctrl dnn_accel_core dnn_reg_if dnn_accel_top` (+ `axil_reg_bridge dnn_accel_axil_top`). The v1 files `accel_ctrl`, `systolic_accel_top` are the matmul-only variant — do not include both tops. |
| Clock | one clock, `clk` (`aclk`); every flop is rising-edge |
| Reset | `rst_n_i` (`aresetn`) asynchronous in → `rst_sync` (2 flops, async assert, sync release) → **synchronous** active-low reset everywhere else. This is the only asynchronous-reset element (the Yosys check shows exactly one `$adff`). |
| CDC | none inside the block. If your SoC bus clock differs from the accelerator clock you must add a CDC (async FIFO or 4-phase handshake) between the AXI slave and `dnn_accel_top`; not provided. |
| Latches | none; combinational blocks assign defaults |
| Parameters | all in `dnn_accel_top` (see chapter 2 §8); derived widths use `$clog2` — do not override them |
| Enable | `busy` gates `lrb`, `sa_core` and `dnn_ppu` (`.en`): when idle the whole datapath holds state. With `set_db lp_insert_clock_gating true` Genus can turn this enable pattern into ICGs. `busy` has a large fan-out (roughly every datapath flop), so give it a buffered tree or register duplicates. |
| Reset on datapath flops | PE operand/psum registers and pipeline stages are reset (X-safety in simulation). It costs area/power: once you trust the valid bits you can remove resets from pure data registers (valid bits and control must keep them). |
| Verilog dialect | Verilog-2001 (`$clog2` only), no SystemVerilog constructs in `rtl/` |

### 3.1 Memories — the biggest area decision

`sram_sdp.v` is a **behavioural, synchronous, one-read/one-write (1R1W)** memory with a registered read output. If synthesized as written, Genus will build every bit from flip-flops. Bits at the default configuration:

| Memory | Words × width | Bits | Ports needed |
|---|---|---|---|
| weight | 512 × 32 | 16 384 | write when idle, read during LDW → can be 1RW or 1R1W |
| activation | 256 × 32 | 8 192 | **1R1W** (LRB feed reads while the PPU writes outputs) |
| bias | 16 × 128 | 2 048 | small: registers are fine |
| accumulator | 32 × 128 | 4 096 | **1R1W** (early read of the next partial sum overlaps a write of the previous) |
| LRB | 8 × 32 | 256 | 1R1W (fill and replay) |
| **total (five SRAMs)** | | **30 976 bits** | |

Recommended: replace weight and activation SRAMs by memory-compiler macros (2-port register-file or 1R1W SRAM), keep bias / accumulator / LRB as flops or small register-file macros. Requirements for any replacement: **synchronous read with 1-cycle latency, read data registered** (the controller and the "early read" accumulate trick depend on it), and no read-during-write dependence on the same address (by construction the design never reads and writes the same address in the same cycle: the DRAIN state waits for every write-back before the next pass starts; the M = 1 / NT = 1 random cases exercise the tightest situations). If only single-port macros are available, the activation and accumulator paths need banking or an extra pipeline stage — that is a design change, not a wrapper change.

Yosys `stat` on the whole top reports about 32 000 memory bits: the additional 1 024 bits are the descriptor array inside `dnn_ctrl` (4 layers × 16 fields × 16 bit), which is also inferred as a small memory (keep it as flops or a tiny register file). Also plan memory BIST and scan (not included).

### 3.2 Likely critical paths (reasoned, not measured)

1. **PE**: 8×8 signed multiply + 32-bit add in one cycle (`psum_out <= psum_in + w × act`). The array replicates it 16×, 256×, … so this path sets the frequency. Options if it fails: split it into multiply / add stages (this lengthens the per-PE hop, so the skew, deskew and `LATENCY` constants in `sa_core` must change together), or use carry-save accumulation.
2. **PPU stage 2**: `sum_r (33 b) × mult (16 b)` — the product register `prod_r` is 50 bits wide (`PW = 33 + 16 + 1`) and the multiply is done in one stage. Splitting it costs one pipeline stage; the controller counts write-backs (`wr_done`) rather than assuming a latency, so no constant has to change.
3. **Accumulate stage**: 4 × 32-bit adders fed by the accumulator-SRAM read data (SRAM clock-to-Q + adder).
4. **Register interface**: 16-bit address decode into the window write enables; wide (128-bit) staging register fan-out.

### 3.3 SDC skeleton (adapt names and numbers to your flow)

```tcl
# ---- clock ---------------------------------------------------------------
set clk_period 10.0                                   ;# start relaxed; tighten after first synthesis
create_clock -name clk -period $clk_period [get_ports clk]        ;# aclk for the AXI top
set_clock_uncertainty 0.2 [get_clocks clk]
set_clock_transition  0.1 [get_clocks clk]

# ---- register-bus I/O (synchronous to clk) -------------------------------
set bus_in  [get_ports {reg_wr_en reg_rd_en reg_addr[*] reg_wdata[*]}]
set_input_delay  -clock clk [expr 0.4*$clk_period] $bus_in
set_output_delay -clock clk [expr 0.4*$clk_period] [get_ports {reg_rdata[*] irq}]

# ---- asynchronous reset ---------------------------------------------------
set_false_path -from [get_ports rst_n_i]              ;# goes only to the synchronizer's async pins
# (or: set_input_delay + recovery/removal via the library; the 2-flop synchronizer handles release)

# ---- environment ----------------------------------------------------------
set_driving_cell -lib_cell BUFX2 [all_inputs]         ;# use your library's cell
set_load 0.05 [all_outputs]

# ---- do NOT add: multicycle paths, extra clocks, generated clocks, case analysis ----
# (none are needed; every path is single-cycle, single clock)
```

For the AXI top replace the bus ports with the `s_axil_*` ports (all synchronous to `aclk`) and `rst_n_i` by `aresetn`. If you insert ICGs (`lp_insert_clock_gating`), Genus creates the gated-clock structure and the SDC needs nothing extra beyond clock-gating checks that Tempus infers.

### 3.4 Getting the number this delivery cannot give you: energy per inference

Cycle counts come from simulation; power comes from your tools. The bridge between them is **switching activity**:

1. Run a representative workload in simulation and dump activity: `bash scripts/run_wave.sh` writes `build/dnn.vcd` for the first batch (weight load + 8 images); for a steadier estimate replay several batches and use only the compute window. Or dump SAIF/TCF with your simulator (Xcelium: `dumpsaif` / `dumptcf`).
2. Synthesis-level power: read it into Genus (`read_vcd` / `read_saif`, then `report_power`) or Joules for RTL power; post-layout: Voltus with SPEF + activity file.
3. **Energy per image** = average power in the compute window × time per image = `P × (cycles_per_image / f_clk)`. At the default configuration `cycles_per_image = 221` (batch 8; 112 at batch 32). Report **pJ per MAC** = energy per image ÷ 1 184 useful MACs (the 64→16→10 network: 64·16 + 16·10 = 1 184, without zero padding).
4. Compare against the byte counts in `00_START_HERE.md`; the data-movement argument predicts where the energy goes (weight/activation/accumulator SRAM vs. PE array) and Voltus' per-hierarchy breakdown will confirm or refute it. Instance names from the top: `u_core.u_wgt_mem`, `u_core.u_act_mem`, `u_core.u_acc_mem`, `u_core.u_lrb`, `u_core.u_ppu`, and the systolic array itself is `u_core.u_core` (an `sa_core`; its PEs are below `u_core.u_core.*`). That comparison is the strongest piece of evidence you can put in the report.

---

## 4. Tool and flow enhancements (short list)

| Idea | Value |
|---|---|
| Replace behavioural SRAMs by macros (§3.1) | realistic area/power; dominates the floorplan |
| Clock gating via `en` (`lp_insert_clock_gating`) | power when idle between batches |
| Operand isolation / zero-gating of the multiplier when `act == 0` | ReLU outputs are often ~40 % zeros (7 of 16 in the traced example) — saves dynamic power; measure with Voltus |
| Multi-Vt / multi-corner sign-off (SS/FF, RC corners) | standard |
| Low-power intent (CPF/UPF) | power-gate the array between batches |
| DFT: scan chains + memory BIST | tape-out requirement |
| Formal (JasperGold) on FSM and register interface | §2 |
| Conformal LEC RTL ↔ netlist | after retiming / ICG insertion |
| Cadence Stratus HLS or Xcelium mixed-language TLM | only if you want to re-model the block at a higher level |

---

## 5. Sign-off checklist for handing the RTL to the back end

- [ ] `bash scripts/check_rtl_quality.sh` prints `RTL QUALITY CHECK PASSED` (lint on both tops + Yosys latch/async scan)
- [ ] `scripts/run_all.sh` prints `ALL STEPS DONE`; `scripts/run_dnn_regress.sh` prints `ALL DNN CONFIGS PASSED`; `scripts/run_mutations.sh` prints `ALL MUTATIONS KILLED`
- [ ] the ASIC hardware parameters (array size, memory depths) are **decided** and identical in: RTL top parameters, `sw/compile.py` arguments (or `sw/dnn_cfg.py` defaults), `-G` overrides in the scripts
- [ ] behavioural SRAMs replaced (or you consciously accept flop-based memories for this tape-out/demo)
- [ ] SDC reviewed; `rst_n_i` false-path only to the synchronizer
- [ ] power-analysis workload chosen (which batch size, which network) and documented
- [ ] known limitations from `02_project_mapping.md` §9 copied into the report's "future work"
