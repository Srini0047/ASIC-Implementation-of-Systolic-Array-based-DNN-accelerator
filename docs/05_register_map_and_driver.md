# 05 — Register Map, Programming Sequences and the C Driver

*This is the contract between software and hardware. If you can write a Verilog testbench that toggles `reg_wr_en`, you can drive this chip; the C driver does the same thing from a CPU.*

All addresses below are **32-bit word addresses** (`reg_addr[15:0]`). Through the AXI4-Lite bridge, byte address = 4 × word address.

---

## 1. Register bus timing

| Signal | Dir | Meaning |
|---|---|---|
| `reg_wr_en` | in | 1-cycle write strobe |
| `reg_rd_en` | in | 1-cycle read strobe |
| `reg_addr[15:0]` | in | word address |
| `reg_wdata[31:0]` | in | write data |
| `reg_rdata[31:0]` | out | read data, valid **one cycle after** `reg_rd_en`, held until the next read |
| `irq` | out | level: `done \| err`; cleared by the next START |

No wait states, no bursts, no byte enables. `reg_wr_en` is sampled at a clock edge. Plain registers (`N_LAYERS`, `ACT_RD_ADDR`) change at that edge; memory-window writes, descriptor writes and the START pulse are registered once inside `dnn_reg_if` and therefore act **one edge later** — nothing reads them in that cycle, so software never sees the difference, but a waveform does.

---

## 2. Register map

### 2.1 Control and status

| Addr | Name | R/W | Bits / meaning |
|---|---|---|---|
| `0x0000` | CTRL | W | bit0 = **START** (write 1; self-clearing pulse) |
| `0x0001` | STATUS | R | bit0 **busy**, bit1 **done**, bit2 **err**, bit3 **rd_valid** |
| `0x0002` | N_LAYERS | RW | number of layers to run, `1 … NUM_LAYERS`. The register is only `⌈log2(NUM_LAYERS+1)⌉` bits wide (3 at the default), so a write of 0 **or of any value above `NUM_LAYERS`** is stored as **0** (it is not truncated — a truncated 9 would silently become 1) and the next START raises `err`. You can read it back to check. |
| `0x0003` | ACT_RD_ADDR | W | activation-SRAM word address to read back; then wait for `rd_valid` |
| `0x0004` | ID | R | `0xD44A0100` — read this first: proves you are talking to the right block |
| `0x0005` | CFG_ARRAY | R | `{ROWS[31:24], M_MAX[23:16], KT_MAX[15:8], NT_MAX[7:0]}` — default `0x04081008` |
| `0x0006` | CFG_LAYERS | R | `NUM_LAYERS` (default 4) |
| `0x0007` | CFG_ACT | R | `ACT_DEPTH` words (256) |
| `0x0008` | CFG_WGT | R | `WGT_DEPTH` (512) |
| `0x0009` | CFG_BIAS | R | `BIAS_DEPTH` (16) |
| `0x000A` | CFG_ACC | R | `ACC_DEPTH` (32) |
| `0x000B` | CFG_LRB | R | `LRB_DEPTH` (8) |
| `0x0010 + b` | ACT_RD_DATA[b] | R | beat *b* of the activation word selected by ACT_RD_ADDR |
| `0x0020` | PERF_CYCLES | R | busy cycles of the last run |
| `0x0021` | PERF_ACT_RD | R | activation-SRAM reads |
| `0x0022` | PERF_LRB_RD | R | LRB (reuse) reads |
| `0x0023` | PERF_WGT_RD | R | weight-SRAM reads |
| `0x0024` | PERF_VEC | R | activation vectors fed to the array |

**STATUS semantics.** `busy`, `done` **and** `err` are made race-free: from the cycle after the START write, `busy = 1`, `done = 0` and `err = 0` are reported even before the internal FSM has moved, so the first STATUS read after a START can never show the result (or the error) of the *previous* run. `err = 1` means a descriptor failed validation (`done` stays 0; `busy` falls to 0). Because `busy` falls in the same cycle as `err` or `done` rises, a poller must test `busy` first: *"while busy → keep polling; then look at err / done"* — this is exactly what the driver's `wait_for()` does. `rd_valid` reads 0 for the one cycle in which the activation-SRAM read that ACT_RD_ADDR triggers is in flight, and 1 otherwise; data is valid from the next cycle on. A bus master faster than one access per cycle must therefore poll `rd_valid` (or wait two cycles) between writing ACT_RD_ADDR and reading ACT_RD_DATA — the driver and the testbench always do.

### 2.2 Layer descriptors — `0x0100 + layer × 16 + field` (write, low 16 bits)

| Field | Name | Range | Meaning |
|---|---|---|---|
| 0 | M | 1 … M_MAX | batch rows (input vectors) for this layer |
| 1 | KT | 1 … KT_MAX | K-tiles = ⌈K / ROWS⌉ |
| 2 | NT | 1 … NT_MAX | N-tiles = ⌈N / COLS⌉ |
| 3 | WBASE | word address | weight-SRAM address of tile (kt = 0, nt = 0) |
| 4 | BBASE | word address | bias-SRAM address of nt = 0 |
| 5 | SRC | word address | activation-SRAM base of this layer's input |
| 6 | DST | word address | activation-SRAM base of this layer's output |
| 7 | MULT | 0 … 65535 | requantization multiplier (unsigned) |
| 8 | SHIFT | 0 … 47 | requantization right shift |
| 9 | RELU | 0 / 1 | 1: clamp to [0, 127]; 0: clamp to [−128, 127] |

Descriptor writes are **ignored while `busy`**. The FSM checks `M`, `KT`, `NT`, `M·NT ≤ ACC_DEPTH`, `SHIFT ≤ 47` and `1 ≤ N_LAYERS ≤ NUM_LAYERS`; a violation sets `err`. **Each layer's descriptor is validated when that layer starts (its FETCH state), not all at once at START**: if layer 2 is bad, layers 0 and 1 have already run and have overwritten their activation regions before `err` is raised — so after an `err`, treat the activation SRAM as garbage and rewrite the inputs. **It does not range-check `WBASE`, `BBASE`, `SRC`, `DST`** — that is the driver's job (`sw/compile.py` and `dnn_load_network()` both do it). A bad base address silently reads or corrupts the wrong memory.

Addresses and shapes follow from these formulas (the same ones the compiler uses):

```
weight word address of tile p = kt·NT + nt, row r :  WBASE + p·ROWS + r
input  word for image m, K-slice kt              :  SRC  + m·KT + kt
output word for image m, N-tile nt               :  DST  + m·NT + nt
bias   word for N-tile nt                        :  BBASE + nt
```

### 2.3 Memory windows

| Window | Base | Word address inside | Write granularity |
|---|---|---|---|
| Weight SRAM | `0x1000` | `0x1000 + word × BSTRIDE_W + beat` | 1 beat = up to 4 int8 lanes |
| Activation SRAM | `0x2000` | `0x2000 + word × BSTRIDE_A + beat` | up to 4 int8 lanes per beat |
| Bias SRAM | `0x3000` | `0x3000 + word × BSTRIDE_B + beat` | **one INT32 per beat**, `COLS` beats per word |

For the default 4×4 array: `BEATS_W = BEATS_A = 1` (4 lanes × 8 bit = 32 bit), `BEATS_B = 4`; strides are the beat counts rounded up to a power of two (1, 1, 4). For an 8×8 array a weight word is 64 bits = 2 beats (stride 2), a bias word is 8 beats (stride 8). **The driver reads `CFG_ARRAY` and derives all of this itself**, so it is not hard-coded.

**Beat rule:** wide words are written LSB beat first; the write of the **last** beat commits the whole word (lower beats wait in a staging register). Lanes are packed little-endian: lane 0 in bits [7:0] of beat 0.

### 2.4 Worked examples (real words from `build/prog.hex`)

**A weight word.** The first weight the compiler writes is row 0 of tile 0 of layer 0: `W1[0, 0..3] = [0, 4, −3, −5]`.

```
lanes (int8):   0x00   0x04   0xFD   0xFB
word (32 bit):  0xFB FD 04 00           (lane 3 in the top byte)
register write: addr 0x1000, data 0xFBFD0400      ← prog.hex line 9:  01 00 1000 fbfd0400
```

**A bias word.** Biases of neurons 0–3 are `[−543, −1544, 441, −71]`; each is a 32-bit two's-complement beat, one per address:

```
0x3000 ← 0xFFFFFDE1   (−543)          0x3002 ← 0x000001B9   (+441)
0x3001 ← 0xFFFFF9F8   (−1544)         0x3003 ← 0xFFFFFFB9   (−71)     ← this last write commits bias word 0
```

**A descriptor** (layer 0 of the default network, base `0x0100`):

| Field | M | KT | NT | WBASE | BBASE | SRC | DST | MULT | SHIFT | RELU |
|---|---|---|---|---|---|---|---|---|---|---|
| value | 8 | 16 | 4 | 0 | 0 | 0 | 128 | 44203 (`0xACAB`) | 25 | 1 |

Layer 1 uses `KT = 4, NT = 3, WBASE = 256, BBASE = 4, SRC = 128, DST = 0, MULT = 48617, SHIFT = 23, RELU = 0`.

---

## 3. Programming sequences

### 3.1 Once, after reset

```
1. reset the block (rst_n_i low ≥ 2 cycles, then high; the synchronizer needs 2 clocks to release)
2. read ID           → must equal 0xD44A0100, else wrong device / wrong address map
3. read CFG_ARRAY, CFG_LAYERS, CFG_ACT, CFG_WGT, CFG_BIAS, CFG_ACC, CFG_LRB
                     → derive ROWS, limits, beat counts, strides
```

### 3.2 Load a network (once per network)

```
for each layer i:
    for kt, nt, r:  write weight word  (WGT_BASE + (WBASE_i + (kt·NT+nt)·ROWS + r)·BSTRIDE_W + beat)
    for nt:         write bias word    (BIAS_BASE + (BBASE_i + nt)·BSTRIDE_B + beat)
    write the 10 descriptor fields     (0x0100 + i·16 + f)
write N_LAYERS
```

### 3.3 Run one batch

```
1. write the M input vectors into the activation window, region SRC of layer 0
2. write CTRL = 1                               (START)
3. wait: poll STATUS until  busy == 0  and  (done | err)      — or wait for irq
      done==1, err==0   success
      err ==1           a descriptor was rejected → fix and START again
4. (optional) read PERF_* counters
5. for each output word (image m, N-tile nt):
      write ACT_RD_ADDR = DST_last + m·NT_last + nt
      poll STATUS.rd_valid
      read ACT_RD_DATA[0 … BEATS_A−1]           → 4 int8 lanes per beat, lane 0 in bits [7:0]
```

Weights and biases stay loaded between runs; only step 1 onward repeats per batch. One START runs **all** layers.

### 3.4 Timing diagram (one run, schematic)

```
clk       ─┐_┌─┐_┌─┐_┌─┐_┌─┐_┌───────────────────────────────┐_┌─┐_┌─┐_┌─
reg_wr_en  ___/‾\_______________________________________________________
CTRL write  START
STATUS      idle(0) │ busy=1 ───────────────────────────────  │ done=1
irq         ______________________________________________/‾‾‾‾‾‾‾‾‾‾‾‾  (level, until next START)
                    └── FETCH → [ LDW → COMP → DRAIN ] × passes → … → IDLE
```

---

## 4. Writing your own register program (no C required)

A *register program* is a text file, one operation per line, 16 hex digits: `OP(2) 00 ADDR(4) DATA(8)`.

| OP | Name | Testbench behaviour |
|---|---|---|
| `01` | WRITE | `reg[ADDR] ← DATA` |
| `02` | WAIT_DONE | poll STATUS until `!busy`; fail unless `done && !err` |
| `03` | READ_EXPECT | read `ADDR`; fail unless equal to `DATA` (value logged) |
| `04` | READ_LOG | read `ADDR`; log only |
| `05` | WAIT_RD_VALID | poll STATUS.rd_valid |
| `06` | DELAY | wait `DATA` clock cycles |
| `07` | WAIT_DONE_ERR | poll until `!busy`; fail unless `err && !done` (negative tests) |
| `FF` | END | stop, print `PLAYER PASSED/FAILED` |

Fastest way to experiment: **`sw/example_tiny_net.py`** builds a 4→4 layer whose answer you can check with a pencil (`y0 = [90, 100, 110, 120] + bias`, ReLU → `[90, 100, 110, 0]`), and `bash scripts/run_tiny.sh` runs it on the RTL. Its last lines (the raw register rows before them look like `R 8 0010 006e645a`: read number, address, data):

```
image 0: RTL returned word 0x006e645a = lanes ['0x5A', '0x64', '0x6E', '0x00'] = [90, 100, 110, 0]
image 1: RTL returned word 0x00030201 = lanes ['0x01', '0x02', '0x03', '0x00'] = [1, 2, 3, 0]
```

The script prints `*** PLAYER PASSED ***` only if these words equal the golden model's `[[90 100 110 0] [1 2 3 0]]` (the program contains `READ_EXPECT` operations). Change `W`, `X`, `BIAS`, `MULT`, `SHIFT`, `RELU` and re-run; try values that saturate (> 127) or a `SHIFT` that halves the result.

---

## 5. The C driver

### 5.1 Files

| File | Purpose |
|---|---|
| `driver/dnn_accel.h` | register constants, types, API |
| `driver/dnn_accel.c` | implementation (portable C99, no libc beyond `string.h`) |
| `driver/hal_mmio.h` | HAL for real memory-mapped hardware (`volatile` pointer) |
| `driver/example_infer.c` | full inference using a *recording* HAL (writes the register trace to a file) |
| `build/net_image.h` | generated by `sw/compile.py`: the quantized network as C arrays (`NET[]`, `NET_X[]`, `NET_Y[]`) |

### 5.2 API

```c
typedef struct {                       /* the only thing you must provide */
    void     *ctx;
    uint32_t (*read )(void *ctx, uint16_t word_addr);
    void     (*write)(void *ctx, uint16_t word_addr, uint32_t data);
    int      (*wait )(void *ctx, int kind);   /* optional; NULL => the driver polls STATUS */
} dnn_hal_t;

int dnn_probe        (dnn_dev_t *d, const dnn_hal_t *hal);                       /* ID + CFG registers        */
int dnn_load_network (dnn_dev_t *d, const dnn_layer_t *layers, int n, int batch);/* weights, bias, descriptors */
int dnn_write_input  (dnn_dev_t *d, const int8_t *x, int k_in);                  /* x[batch][k_in], zero-padded */
int dnn_run          (dnn_dev_t *d);                                             /* START + wait done          */
int dnn_read_perf    (dnn_dev_t *d, dnn_perf_t *p);
int dnn_read_output  (dnn_dev_t *d, int8_t *y);                                  /* y[batch][np_last]          */
int dnn_argmax       (const int8_t *row, int n);
```

Return codes: `DNN_OK`, `DNN_E_ARG` (bad call order/arguments), `DNN_E_ID` (wrong device), `DNN_E_CAPACITY` (network does not fit this hardware — nothing has been written yet), `DNN_E_HW` (hardware raised `err`), `DNN_E_TIMEOUT`.

`dnn_load_network()` **plans the whole network in software first** (padding, tile counts, memory bases, all limit checks) and only then starts writing registers, so a network that does not fit leaves the chip untouched. Weights are given padded (`kp`, `np` multiples of `ROWS`) — the Python compiler already did that.

### 5.3 Typical use (bare metal, memory-mapped)

```c
#include "dnn_accel.h"
#include "hal_mmio.h"
#include "net_image.h"                              /* NET[], NET_X[], ... from sw/compile.py */

dnn_mmio_t mmio = { (volatile uint32_t *)0x40000000u };   /* your AXI-Lite base address */
dnn_hal_t  hal  = { &mmio, dnn_mmio_read, dnn_mmio_write, NULL };
dnn_dev_t  dev;
int8_t     y[NET_BATCH * 12];

dnn_probe(&dev, &hal);
dnn_load_network(&dev, NET, NET_LAYERS, NET_BATCH);      /* once */
dnn_write_input(&dev, NET_X, NET[0].kp);                 /* per batch */
dnn_run(&dev);
dnn_read_output(&dev, y);
int digit = dnn_argmax(&y[0], 10);
```

`hal_mmio.h` is `base[word_addr]` on a `volatile uint32_t *` — the word index equals the byte offset ÷ 4, which is exactly the AXI byte-address scaling. **Not tested on real hardware** (there is none); the same HAL structure is used by the recorder that *is* tested.

**Linux user space:** `mmap()` the region (`/dev/mem`, or a UIO/VFIO device) and use the returned pointer as `base`. **Interrupts:** implement `hal.wait` with `poll()`/`select()` on the UIO file descriptor instead of polling STATUS.

### 5.4 How the driver is verified

```bash
bash scripts/check_driver.sh
```
1. compiles `dnn_accel.c` + `example_infer.c` with `-std=c99 -Wall -Wextra -Wpedantic -Werror`;
2. runs it — the recorder HAL writes all 568 register accesses of "probe + load + first batch" to `build/c_trace.hex` (and answers the few reads the driver performs: ID/CFG registers, and the expected output words);
3. **`cmp` against the first 568 lines of the Python-generated `build/prog.hex`** — must be byte-identical (two independent implementations of the same protocol agree);
4. replays the C-generated stream on the RTL in Verilator — must give `PLAYER PASSED`.

To prove this check has teeth, `bash scripts/run_mutations.sh` (group C) injects four deliberate bugs into a copy of the driver — descriptor SRC/DST swapped; weight rows in wrong order; bias lanes reversed; readback addresses strided wrongly — and requires each to be caught by the stream comparison or the output check.

**What this does *not* cover:** the recorder HAL supplies its own `wait()`, so the STATUS *polling loop* inside `wait_for()` and `hal_mmio.h` are only compile-checked (`-Werror`), not executed against the RTL. The loop is short and mirrors the polling the testbench player performs, but if you port the driver, test that path on your first real bus access (e.g. read STATUS after START and confirm you see `busy` and no stale `err`).

### 5.5 Porting checklist

1. Implement `read`/`write` for your bus (or use `hal_mmio.h`).
2. Make sure stores are not reordered or merged (`volatile`, device-memory attributes, barriers before the START write).
3. Call `dnn_probe()` and **check the return code** — the ID read is your first proof the address map is right.
4. Make sure `ROWS`, limits etc. in `net_image.h` (from `sw/compile.py`) match the silicon you built; `dnn_load_network()` re-checks against what the hardware reports and returns `DNN_E_CAPACITY` if not.

---

## 6. The AXI4-Lite bridge

`rtl/axil_reg_bridge.v`, wrapped by `rtl/dnn_accel_axil_top.v`. Handles AW/W/B and AR/R handshakes one transaction at a time.

| Property | Behaviour |
|---|---|
| Address | `awaddr[17:2]` / `araddr[17:2]` → `reg_addr` (256 KiB window) |
| Write | accepted when **both** `AWVALID` and `WVALID` are high (`AWREADY = WREADY = idle & AWVALID & WVALID`), then `reg_wr_en` for one cycle and `BVALID` |
| Read | `ARREADY` when idle and no write pending; `reg_rd_en`; `RVALID` two cycles later with `RDATA = reg_rdata` held stable until `RREADY` |
| Responses | always `OKAY` (`00`); `WSTRB` ignored (full-word writes), `AxPROT` ignored |
| Priority | write over read if both are pending |

Verified by replaying the entire digit-network program and the random regression program through the bridge with **randomized master behaviour**: AW and W presented at different times, gaps between requests, `BREADY`/`RREADY` delayed. Results are identical to the direct register-bus run. Four bridge mutations (early `RVALID`, `AWREADY` without `WVALID`, `WREADY` without `AWVALID`, wrong address slice) are injected by `scripts/run_mutations.sh` (group D) and were all caught. Note the AXI runs replay the Python-generated program (`prog.hex`); the C driver's stream was proven identical to it for the first batch (§5.4), so the two claims combine transitively rather than by a single run.

---

## 7. Error cases and recovery

| Situation | Symptom | Fix |
|---|---|---|
| `M`, `KT`, `NT` zero or above the hardware limit; `M·NT > ACC_DEPTH`; `SHIFT > 47`; `N_LAYERS` 0 or above `NUM_LAYERS` (reads back as 0) | `STATUS.err = 1`, `done = 0`, `irq` — raised when the offending layer starts, so earlier layers may already have run | correct the descriptor / count, rewrite the inputs, START again (a new START clears `err`) |
| `WBASE`/`BBASE`/`SRC`/`DST` outside the memories | **no error flag** — silent wrong results | your driver must check (both provided software paths do) |
| Descriptor, weight or bias written while busy | ignored (the activation SRAM write port belongs to the post-processing unit while busy, so host writes there are lost too) | wait for done |
| Reading an output before `rd_valid` | stale data | always wait for `rd_valid` after writing `ACT_RD_ADDR` |
| Wrong `ID` | not our chip / wrong base address | check address decode |
