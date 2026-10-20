# 02 — What You Are Building, Block by Block

*Chapter 1 gave you the vocabulary. This chapter turns the report's goals into concrete hardware, walks one batch of 8 images through it cycle by cycle, and tells you which file implements each idea.*

---

## 1. The problem, in two sentences

**What you're building:** a small silicon block that takes an image, runs it through a trained two-layer neural network entirely by itself (integer arithmetic only), and hands back ten scores — the CPU only loads the network once, writes the image in, presses START, and reads ten numbers out.

**What problem it solves:** on a CPU this pattern (millions of multiply-accumulates, each needing two operand fetches) wastes almost all its energy moving data. The block is arranged so that each weight, each activation and each partial sum is fetched **as few times as possible** and travels between **neighbouring registers** rather than through a memory hierarchy.

Your report's title — *ASIC Implementation of a Systolic Array Based DNN Accelerator* — maps onto that as follows.

| Report goal | Hardware building block | File(s) |
|---|---|---|
| "INT8 MAC processing element" | **PE**: 8-bit weight register, 8×8 signed multiplier, 32-bit adder, two output registers | `rtl/pe.v` |
| "Systolic array" | **4×4 mesh of PEs** + input skew + output deskew | `rtl/systolic_array.v`, `rtl/skew_delay.v`, `rtl/sa_core.v` |
| "Weight-stationary dataflow" | weights shift into PE registers once per tile; the batch streams through | `rtl/pe.v` (`weight_load`), `rtl/dnn_ctrl.v` (`S_LDW`) |
| "Local Reuse Buffer" | small buffer replaying activation vectors so the big SRAM is read once | `rtl/lrb.v` |
| "Hierarchical on-chip memory" | SRAM → LRB → PE registers (+ accumulator SRAM) | `rtl/sram_sdp.v` (×4), `rtl/lrb.v` |
| "Control logic" | **FSM** that sequences layers, tiles, load/compute/drain | `rtl/dnn_ctrl.v` |
| *(DNN emphasis, added in v2)* accumulate over K, bias, requantize, ReLU | **accumulate stage + PPU** | `rtl/dnn_accel_core.v`, `rtl/dnn_ppu.v` |
| *(added in v2)* run a whole network | layer **descriptors**, ping-pong activation regions | `rtl/dnn_ctrl.v`, `rtl/dnn_reg_if.v` |
| *(added in v2)* talk to a CPU | register interface (+ AXI4-Lite bridge) | `rtl/dnn_reg_if.v`, `rtl/axil_reg_bridge.v` |
| Performance metrics | hardware performance counters | `rtl/perf_counters.v` |

---

## 2. The whole chip on one page

```
                                   dnn_accel_axil_top   (SoC-ready top)
   ┌────────────────────────────────────────────────────────────────────────────────────────────┐
   │                                                                                            │
   │  CPU ──AXI4-Lite──► axil_reg_bridge ──reg bus──► dnn_accel_top                             │
   │                                                                                            │
   │      ┌─────────────────────────── dnn_accel_top ───────────────────────────────────┐       │
   │      │ rst_sync                                                                    │       │
   │      │                                                                             │       │
   │      │ dnn_reg_if ── window writes ──►  weight SRAM   512 × 32 b  (2 KiB)          │       │
   │      │  (register map,   ───────────►  bias SRAM      16 × 128 b                   │       │
   │      │   beat staging)   ◄──────────►  activation SRAM 256 × 32 b (1 KiB, 2 halves)│       │
   │      │       │                                                                     │       │
   │      │       └── descriptors, START, N_LAYERS ──►  dnn_ctrl  (FSM)                 │       │
   │      │                                              │ control strobes, addresses   │       │
   │      │                                              ▼                              │       │
   │      │     activation SRAM ──► LRB ──► sa_core ─► accumulate ─► PPU ─► act SRAM    │       │
   │      │     (source half)   (replay)   skew→4×4 PEs→deskew    ▲  │     (dest half) │       │
   │      │                                                       │  ▼                  │       │
   │      │                                                  accumulator SRAM           │       │
   │      │                                                  32 × 128 b (partial sums)  │       │
   │      └─────────────────────────────────────────────────────────────────────────────┘       │
   └────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Analogy — a small factory.** The **weight SRAM** is the tool store. The **PEs** are 16 workers at fixed stations, each holding one tool (its weight). The **activation SRAM** is the warehouse of raw material; the **LRB** is the notepad by the line so the foreman does not walk to the warehouse for material he needs again. Material moves left-to-right on one belt, running totals move top-to-bottom on another. The **accumulator SRAM** is the shelf where half-finished totals wait while the workers are re-tooled for the next slice of the job. The **PPU** is the quality-control desk: it adds the bias, rescales the 32-bit total down to one byte, clips negatives (ReLU) and puts the finished byte in the other half of the warehouse. The **FSM** is the foreman, and each **layer descriptor** is his work order.

### Module list (top-down)

| Module | Job | Storage / size at defaults |
|---|---|---|
| `dnn_accel_axil_top` | AXI4-Lite pins + `irq` | — |
| `axil_reg_bridge` | AXI4-Lite slave → simple register bus | 2-bit FSM |
| `dnn_accel_top` | reset synchronizer + register interface + core | — |
| `rst_sync` | async-assert / sync-release reset — **the only asynchronous element** | 2 flops |
| `dnn_reg_if` | register map; packs 32-bit bus beats into wide SRAM words; STATUS | staging regs |
| `dnn_ctrl` | layer/tile sequencer; validates descriptors; sets `err` | descriptor regs 4 layers × 16 × 16 b |
| `sram_sdp` × 4 | behavioural 1-read/1-write synchronous SRAM (weight, bias, activation, accumulator) | 2 KiB / 256 B / 1 KiB / 512 B |
| `lrb` | Local Reuse Buffer + source mux + feed register | 8 × 32 b |
| `sa_core` | input skew, PE mesh, output deskew, valid pipeline | 16 PEs |
| `pe` (×16) | weight-stationary INT8 MAC | 8 + 8 + 32 b registers each |
| (accumulate stage in `dnn_accel_core`) | `acc = res + (first K-tile ? 0 : stored partial)` | 1 pipeline reg |
| `dnn_ppu` | 3-stage bias + requantize + ReLU/clamp | 4 lanes |
| `perf_counters` | busy cycles, activation-SRAM reads, LRB reads, weight-SRAM reads | 4 × 32 b (the fifth register, PERF_VEC = vectors fed to the array, is a counter `vec_cnt` inside `dnn_accel_core`) |

---

## 3. What one PE does, and how 16 of them compute a matrix product

Each clock, every PE does:

```
act_out  <= act_in                              activation moves RIGHT one PE
psum_out <= psum_in + weight_reg × act_in       running total moves DOWN one PE
(weight_load) weight_reg <= weight_in           weights shift DOWN when loading
```

For a `4×4` tile of weights `W[r][c]` sitting in the array and one input vector `a[0..3]`, column `c`'s bottom output is

```
out[c] = a[0]·W[0][c] + a[1]·W[1][c] + a[2]·W[2][c] + a[3]·W[3][c]
```

— four MACs done by the four PEs of that column, in a wave. Row `r` of the input must arrive `r` cycles late so that it meets the running total coming from row `r−1`; that stagger is the **skew** (and `sa_core` un-staggers the outputs, the **deskew**). A vector entering at cycle τ has all four results aligned at τ + ROWS + COLS − 1 = **τ + 7**.

The v1 notes (`DESIGN_NOTES.md` §3.4) contain a cycle-by-cycle 2×2 trace (`tb/trace_2x2.v`) of exactly this — read it once, and the array will never seem mysterious again.

**The array only ever sees a 4-wide slice of the input (K = 4) and produces 4 outputs (N = 4).** A real layer is bigger (`K = 64`, `N = 16`), so the controller must cut the problem into tiles and stitch them. That is the job of the FSM.

---

## 4. Tiling: how a 64×16 layer runs on a 4×4 array

`Y[8×16] = X[8×64] · W[64×16]`, `ROWS = COLS = 4`:

```
              N (16 outputs) = 4 N-tiles of 4 columns
            ┌────┬────┬────┬────┐
   K=64 =   │ 0,0│ 0,1│ 0,2│ 0,3│   K-tile 0   (rows 0–3 of W)
   16 K-    │ 1,0│ 1,1│ 1,2│ 1,3│   K-tile 1
   tiles    │  … │  … │  … │  … │
            │15,0│15,1│15,2│15,3│   K-tile 15
            └────┴────┴────┴────┘
```

The array can hold **one** tile at a time. So: `16 × 4 = 64 passes`, and for each pass:

1. **LDW** (load weights, 4 cycles): shift the 4×4 tile into the PEs from the weight SRAM.
2. **COMP** (compute, 8 cycles): stream the 8 images' K-slices through the array, one vector per cycle.
3. **DRAIN** (≈ 11 cycles on average): wait for the pipeline to empty; results are written or accumulated as they come out.

Loop order in `dnn_ctrl.v` (this order is a design decision — see "why kt outer" below):

```
for layer in 0 .. n_layers-1:              S_FETCH: read this layer's descriptor, validate it
  for kt in 0 .. KT-1:                     K slice (outer)
    for nt in 0 .. NT-1:                   N tile (inner)
      S_LDW   : load weight tile (kt, nt)                            ROWS cycles
      S_COMP  : for m in 0..M-1: feed act[m][kt] to the array       M cycles
                  nt == 0 : read from activation SRAM AND copy into the LRB
                  nt  > 0 : replay from the LRB  (no SRAM read)
      S_DRAIN : results of pass arrive:
                  kt == 0        : acc[m][nt]  = result
                  0 < kt < last  : acc[m][nt] += result
                  kt == last     : (acc[m][nt] + result + bias) → PPU → activation SRAM (other half)
```

**Why K-tile outer, N-tile inner?** The same K-slice `act[·][kt]` (8 vectors) is needed for *every* N-tile. Keeping `kt` outer means those 8 vectors are used 4 times in a row: read from the SRAM on the first N-tile, replayed from the LRB on the other three. That is precisely what the LRB was designed for. The price is that partial sums for the whole `[8×16]` output must wait in the accumulator SRAM between K-tiles (32 words × 128 bit, i.e. `M × NT = 8 × 4`). The alternative (N outer, K inner) would keep partial sums in a register but re-read activations 4× from SRAM. I did not build or measure that order; it is listed as roadmap item 7.11 (`04_roadmap.md`) if you want the comparison.

### The accumulate stage and the "one cycle early" trick

The accumulator SRAM has 1-cycle read latency. To add a stored partial sum to a result the very cycle the result appears, the read must be issued one cycle **before** the result is valid. `sa_core` therefore exports `res_vld_early` (the valid pipeline tapped one stage earlier). `acc_rd_en = res_vld_early & ~first_kt`, then `sum = res + (first_kt ? 0 : acc_rd_data)`. Whether this is right is *verified*, not argued: the random regression covers K-tile counts from 1 to 16 and would fail on any off-by-one.

### The PPU (post-processing unit)

Three registered stages, four lanes in parallel (`rtl/dnn_ppu.v`):

```
stage 1:  sum_r  = acc + bias                              (ACC_WIDTH+1 bits)
stage 2:  prod_r = sum_r × mult                            (mult: 16-bit unsigned)
stage 3:  q_r    = clamp( (prod_r + 2^(shift-1)) >>> shift , lo , 127 )     lo = ReLU ? 0 : -128
```

`mult` and `shift` are per layer (in the descriptor). This is *exactly* the arithmetic of `sw/golden.py` — that equality is what "bit-exact" means.

---

## 5. Memories: what lives where (default configuration)

| Memory | Word | Depth | Size | Holds | Written by | Read by |
|---|---|---|---|---|---|---|
| Weight SRAM | 4 × INT8 = 32 b | 512 | 2 KiB | all layers' weight tiles, in pass order | host | `dnn_ctrl` during LDW |
| Bias SRAM | 4 × INT32 = 128 b | 16 | 256 B | one word per N-tile | host | `dnn_ctrl` at the last K-tile |
| Activation SRAM | 4 × INT8 = 32 b | 256 | 1 KiB | **two halves** of 128 words: input image(s) / layer outputs | host (input), PPU (outputs) | LRB feed, host (result readback) |
| Accumulator SRAM | 4 × INT32 = 128 b | 32 | 512 B | K-tile partial sums, `[M × NT]` | accumulate stage | accumulate stage |
| LRB | 4 × INT8 = 32 b | 8 | 32 B | the current K-slice's activation vectors | fill on `nt == 0` | replay on `nt > 0` |
| PE registers | | | 16 × 8 b weights | current weight tile | shift-load | multiplier |

These are **behavioural** SRAMs (`sram_sdp.v`). In your ASIC flow they should be replaced by memory-compiler macros; if you synthesize them as written, Genus will build them from flip-flops (30 976 bits in the five memories at the defaults), which is large. See `06_verification_and_backend.md` §3.1.

### Ping-pong regions of the activation SRAM

Layer *i* reads its inputs from one half and writes its outputs into the other half; layer *i+1* swaps roles. Nothing is copied; the descriptors just carry different `SRC`/`DST` bases (0 or `M_MAX × KT_MAX` = 128). Layer 0 reads region 0, writes region 1; layer 1 reads region 1, writes region 0; the host reads the final result from wherever the last layer wrote. The RTL requires `ROWS == COLS` because a layer's output tile (COLS wide) must be the next layer's input slice (ROWS wide).

### Address map of the default network (from `sw/compile.py`)

```
layer 0: K 64→64 (KT=16)   N 16→16 (NT=4)   passes = 64   src = 0    dst = 128   wbase = 0
layer 1: K 16→16 (KT= 4)   N 10→12 (NT=3)   passes = 12   src = 128  dst = 0     wbase = 256
```

* Weight SRAM: layer 0 occupies words 0–255 (64 tiles × 4 rows), layer 1 words 256–303. Total 304 of 512.
* Word address of tile `p = kt·NT + nt`, row `r` is `wbase + p·ROWS + r`.
* Input word for image `m`, K-slice `kt` is `SRC + m·KT + kt`. Output word for image `m`, N-tile `nt` is `DST + m·NT + nt`.
* `N = 10` is padded to 12 (three tiles of four) with zero weights and zero bias; the host ignores lanes 10–11.

---

## 6. A cycle-level walk through one pass and one output tile

**One pass** (4×4, M = 8):

| Phase | Cycles | What the hardware does |
|---|---|---|
| LDW | 4 | `wgt_rd_en`; the weight SRAM streams 4 words (tile row 3 first), each shifting down the columns |
| COMP | 8 | one activation vector per cycle enters the skew network; `nt == 0` also fills the LRB |
| DRAIN | ≈ 11 (avg. 11.3) | array latency (`ROWS + COLS − 1 = 7`) plus register stages; last results are accumulated / requantized / written |

Measured on the default network (one batch of 8, from an instrumented simulation): **76 passes, 1 771 cycles** — FETCH 2, LDW 304 (17 %), **COMP 608 (34 %)**, DRAIN 857 (48 %). Only a third of the cycles are the array streaming real data. That is the main inefficiency, and the biggest speed-up opportunity (§8).

**Life of output tile `(m = 0, nt = 0)` of layer 0** — 16 passes touch it:

```
pass (kt=0 ,nt=0): result r0            → acc[0][0] = r0                       (write accumulator SRAM)
pass (kt=1 ,nt=0): result r1            → acc[0][0] = acc[0][0] + r1           (read-modify-write)
   …
pass (kt=14,nt=0): result r14           → acc[0][0] += r14
pass (kt=15,nt=0): result r15           → v = acc[0][0] + r15 + bias[0..3]
                                          → PPU: (v × mult + round) >> 25, clamp [0,127]
                                          → activation SRAM, destination half, word 128 + 0·4 + 0
```

Fifteen partial-sum writes and fifteen reads for every output word: that traffic is what dominates the bytes-per-MAC figure in `00_START_HERE.md`.

---

## 7. The register interface in one paragraph

The whole chip is driven by a 32-bit register bus: `reg_wr_en`, `reg_rd_en`, `reg_addr[15:0]` (word address), `reg_wdata`, `reg_rdata`. Small registers (CTRL, STATUS, N_LAYERS, ID, config, performance counters) sit at low addresses; layer descriptors at `0x100 + layer×16 + field`; and three *windows* map the big memories: weights at `0x1000`, activations at `0x2000`, biases at `0x3000`. A memory word wider than 32 bits is written in **beats** (e.g., a 128-bit bias word = four 32-bit writes; the last beat commits). A CPU sees this as a block of memory-mapped registers. Full map: `05_register_map_and_driver.md`.

---

## 8. Configuration knobs and their constraints

| Parameter | Default | Meaning | Constraint |
|---|---|---|---|
| `ROWS` = `COLS` | 4 | array size (K-tile and N-tile width) | must be equal; random regression covers 1, 2, 3, 4, 8; the digit network was also run at 16 |
| `M_MAX` | 8 | largest batch per run | `M ≤ M_MAX` |
| `KT_MAX`, `NT_MAX` | 16, 8 | largest K-tiles / N-tiles per layer | `K ≤ ROWS·KT_MAX`, `N ≤ ROWS·NT_MAX` |
| `ACT_DEPTH` | 256 | activation words | `≥ 2 · M_MAX · KT_MAX` (two halves) |
| `WGT_DEPTH` | 512 | weight words | `Σ layers KT·NT·ROWS ≤ WGT_DEPTH` |
| `BIAS_DEPTH` | 16 | bias words | `Σ layers NT ≤ BIAS_DEPTH` |
| `ACC_DEPTH` | 32 | accumulator words | `M · NT ≤ ACC_DEPTH` per layer |
| `LRB_DEPTH` | 8 | vectors cached | any ≥ 1; if `M > LRB_DEPTH` only the first `LRB_DEPTH` vectors are reused |
| `NUM_LAYERS` | 4 | descriptor slots | ≤ 16 (C driver limit) |
| `ACC_WIDTH` | 32 | accumulator bits | register bus assumes ≤ 32 |

`sw/compile.py` checks every constraint before emitting a program and names the violated one; the RTL independently checks the descriptor ranges and raises `err` (interrupt) if they are violated — this is exercised by the negative tests in the regression: `M`, `KT`, `NT` = 0 or max+1, `M·NT > ACC_DEPTH`, `SHIFT` = 48, and `N_LAYERS` = 0, `NUM_LAYERS+1` or a wrapping value (see `06_…` §1).

### How the array size changes the picture (measured — `python3 sw/sweep_arch.py`)

Same trained network (64→16→10), each row a separate RTL build and run. Every row is a hardware instance resized to fit its batch — `LRB_DEPTH = batch`, `ACC_DEPTH = batch·NT`, `ACT_DEPTH = 2·batch·KT` — so the LRB can hold a whole batch; the first row is the default configuration. Only 3 batches (24 images at batch 8, 96 at batch 32) are simulated per row and the per-batch counters are read from the last one; cycle counts do not depend on pixel values, so this is enough. Weight and activation bytes come from the RTL's performance counters; the partial-sum bytes come from the tiling formula `(KT−1)·M·NT` words written and read per layer, and the script **fails if that formula differs from the RTL's real accumulator-SRAM strobe counts** (it did not, for any row):

| Array | Batch | K-tiles | cycles/image | useful MAC/cycle | utilization | on-chip bytes per useful MAC |
|---|---|---|---|---|---|---|
| 4×4 | 8 | 16 | 221.4 | 5.35 | 33.4 % | 2.250 |
| 4×4 | 32 | 16 | 112.3 | 10.54 | 65.9 % | 2.154 |
| 8×8 | 8 | 8 | 89.2 | 13.27 | 20.7 % | 1.135 |
| 8×8 | 32 | 8 | 37.3 | 31.73 | 49.6 % | 1.034 |
| 16×16 | 32 | 4 | 13.2 | 89.57 | 35.0 % | 0.426 |

Read the table as the design guide it is:

* **Bigger batch → better utilization** (fill/drain overhead is amortized over more vectors).
* **Bigger array → fewer K-tiles → far fewer partial-sum spills** (bytes per MAC falls 5×), at the price of lower utilization on a network this tiny.
* A **16×16 array on a 64→16→10 network** is oversized; match the array to the layers you intend to run.

---

## 9. Known limitations (so nobody is surprised)

| Limitation | Consequence | Fix (roadmap phase 7) |
|---|---|---|
| Weight load, compute, drain run one after another | only 34 % of cycles are streaming; upper bound of a perfect overlap ≈ 2.9× | double-buffer weights; overlap drain with next tile |
| Partial sums go through an SRAM read-modify-write | dominates bytes/MAC at 4×4 | larger array, output-stationary accumulation, or wider accumulate registers |
| Fully-connected layers only | no convolution / pooling / batch-norm hardware | lower convolution to a GEMM in software (im2col) |
| Only ONNX `Gemm` (+`Relu`) supported by the software flow | anything else stops with a clear message | extend `sw/quantize.py` |
| Symmetric per-tensor INT8 | slightly lower accuracy on harder networks | per-channel scales |
| Batch fixed per network load | changing the batch size means rewriting descriptors | trivial software change |
| Behavioural SRAMs | area/power/timing unknown | memory macros in your flow |
| No power, area or timing numbers | I have not run Genus/Innovus/Voltus | your back-end flow (`06_…`) |

Next: **`03_software_stack.md`** — every tool from PyTorch to the register bus.
