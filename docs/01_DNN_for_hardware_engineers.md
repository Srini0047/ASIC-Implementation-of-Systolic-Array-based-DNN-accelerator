# 01 — Deep Neural Networks for a Hardware Engineer

*You know RTL. You know nothing about machine learning. This chapter gives you exactly the ML you need — and nothing else — in the vocabulary you already have: registers, memories, adders, multipliers, bandwidth.*

Every number below comes from this project's own files. You can reproduce all of them:

```bash
python3 sw/trace_one.py 0 5      # follow one image through the network, printing every number
bash scripts/run_all.sh          # the whole flow
```

---

## 1. The 60-second version

A neural network used for **inference** (the only thing our chip does) is a **fixed, feed-forward datapath**:

```
input numbers ──► [ layer 1 ] ──► [ layer 2 ] ──► … ──► output numbers
                   (multiply-accumulate, then a cheap squash)
```

* There is **no branching, no data-dependent control flow, no pointer chasing.** The sequence of operations is identical for every input. That is what makes it such a good fit for hardware.
* Everything is **multiply-accumulate (MAC)**: `acc = acc + a × b`. More than 99 % of the arithmetic in a network like this is that one operation.
* The "program" is a big pile of **constants (weights)** that were found *offline* by a training program on a PC. The chip never learns anything; it just evaluates the frozen function.

**Analogy — a voting committee.** A *neuron* is a committee that looks at many inputs. Each input gets a vote weighted by how much this committee cares about it (the **weight**). The committee sums the weighted votes, adds a fixed nudge (the **bias**), and if the total is negative it stays silent (**ReLU**: negative → 0). A *layer* is 16 committees all looking at the same 64 inputs. A *network* is layers chained: the outputs of one layer's committees are the inputs of the next.

---

## 2. From maths to hardware, step by step

### Step 1 — One neuron is one dot product = a chain of MACs

```
acc = bias;
for k in 0..63:            // 64 inputs
    acc = acc + x[k] * w[k];     // one MAC per input
y = (acc < 0) ? 0 : acc;         // ReLU  (a comparator + mux: essentially free)
```

In RTL terms: a multiplier feeding an adder feeding a register. Nothing more. (Our `rtl/pe.v` is exactly this, with the weight held in a register.)

### Step 2 — One layer = many neurons sharing the same inputs = a matrix product

16 neurons × 64 weights each = a **weight matrix** `W` of 64 rows × 16 columns (column *n* holds neuron *n*'s weights). The layer computes, for input vector `x` (64 numbers):

```
y = ReLU( x · W + b )        shapes:  [1×64] · [64×16] + [16] → [1×16]
```

Process several images at once (a **batch** of `M`) and the input becomes a matrix; this is a **matrix-matrix multiply (GEMM)**:

```
Y[M×N] = X[M×K] · W[K×N]      M = batch, K = inputs per neuron, N = neurons
```

Only three integers — **M, K, N** — describe the whole workload of a layer. Memorize them; the rest of the project is about mapping `(M, K, N)` onto a small grid of MAC units.

### Step 3 — Where the numbers live

| Word | What it is (hardware view) | Changes when? | Where it lives on our chip |
|---|---|---|---|
| **Weights** | constants, one per (input, neuron) pair | never at inference time (loaded once) | weight SRAM → PE weight registers |
| **Bias** | one constant per neuron | never | bias SRAM |
| **Activations** | the data flowing between layers (input image, layer-1 outputs, …) | every inference | activation SRAM (+ reuse buffer) |
| **Partial sums** | half-finished dot products | every cycle | PE registers, accumulator SRAM |
| **Scale factors** | how to convert between integers and real numbers | never | folded into `(mult, shift)` (chapter 3) |

**Tensor** is just the ML word for "N-dimensional array". `[64×16]` is a 2-D tensor (a matrix), `[8×64]` a batch of 8 images. It is memory with an indexing convention, nothing more. **Shape** is the list of dimensions.

Careful with one confusing bit of vocabulary: **"activation"** means *the data value* that comes out of a layer; **"activation function"** means *the squashing step* (ReLU) applied to it. Two different things, unfortunately sharing a name.

### Step 4 — Why the squash (ReLU) exists

Without ReLU, `layer2(layer1(x))` = `(x·W1)·W2` = `x·(W1·W2)` — the whole network would collapse into one matrix product and could only learn straight-line boundaries. ReLU's bend at zero is what lets a stack of layers represent complicated functions. For us it costs one comparator per output. We even fold it into the final saturation step (clamp lower bound 0 instead of −128).

### Step 5 — Training vs. inference

| | Training | Inference (our chip) |
|---|---|---|
| Runs | once, offline, on a PC/GPU, for minutes–weeks | every time someone needs an answer |
| Does | searches for the weights that minimize error (heavy, needs backward pass, floating point) | computes one forward pass with frozen weights |
| Output | weight file | class label / score |

You never have to understand back-propagation to build this accelerator. You only need to know that training produces the weight tensors and that they are then treated as **constants**.

---

## 3. One image, traced with real numbers

The network in this project classifies 8×8 hand-written digit images (scikit-learn "digits" data set). It has two layers: **64 → 16 (ReLU) → 10**. Running `python3 sw/trace_one.py 0 5` prints this (trimmed):

```
Step 0 - input: an 8x8 image, pixel values 0..16                  shape [1, 64] after flattening
    0  0  2 13 16  9  0  0
    0  0 12 12  7 16  3  0          quantized to int8: x_q = round(x / s_in), s_in = 0.00787
    ...                             x_q[0:12] = [0, 0, 16, 103, 127, 71, 0, 0, 0, 0, 95, 95]

Step 1 - layer 1, neuron 5  (one column of W1[64,16]):  64 multiplies, 63 adds
      k= 2:  x_q[2] * W1[2,5] =   16 *   -7 =   -112
      k= 3:  x_q[3] * W1[3,5] =  103 *  -16 =  -1648
      k= 4:  x_q[4] * W1[4,5] =  127 *  -10 =  -1270          ... 59 more products
   sum of 64 products (INT32 accumulator)  = 29657
   + bias[5]                               = 29657 + 4550 = 34207
   x mult (44203)                          = 1512052021
   + rounding, then >> 25                  = 45
   ReLU / clamp to [0,127]                 = 45      <- neuron 5's INT8 output

   all 16 neurons (int8): [0,0,0,0,0,45,32,0,2,69,23,40,38,7,0,39]     shape [1,16]
   -> 7 of 16 are exactly 0: that is what ReLU did

Step 2 - layer 2: 16 -> 10 scores (W2 is [16,10], no ReLU)
   scores (int8): [-8, -14, -24, -10, -3, -27, -34, 34, -4, 20]       shape [1,10]
   argmax = 7   (true digit 7)   CORRECT
```

The **shape story** for a batch of 8 images:

| Stage | Tensor | Shape | Element type | MACs |
|---|---|---|---|---|
| input | X | [8 × 64] | int8 | |
| layer 1 weights | W1 | [64 × 16] | int8 | |
| layer 1 accumulators | X·W1 + b1 | [8 × 16] | **int32** (a sum of 64 products of two int8 needs ~22 bits, we keep 32) | 8·64·16 = 8 192 |
| layer 1 output | H | [8 × 16] | int8 (after bias, requantize, ReLU) | |
| layer 2 weights | W2 | [16 × 10] | int8 | |
| layer 2 accumulators | H·W2 + b2 | [8 × 10] | int32 | 8·16·10 = 1 280 |
| network output | scores | [8 × 10] | int8 → `argmax` gives the digit | |

Per image that is **1 184 MACs** and **1 184 weight bytes**. That is small on purpose (fast to simulate). For scale (approximate, widely quoted figures): MobileNet-class networks are a few hundred million MACs per image, ResNet-50 about four billion. Exactly the same datapath, just more passes.

**Exercise (2 minutes).** How many MACs and weights does a 784 → 128 → 10 network (the classic MNIST MLP) need? *Answer: 784·128 + 128·10 = 100 352 + 1 280 = 101 632 per image.*

---

## 4. Why the numbers are integers (quantization, in one page)

A trained network is stored in 32-bit floating point (fp32). An fp32 multiplier is large, slow and hungry. Since a well-trained network tolerates small errors, we replace each real number by a small integer times a per-tensor **scale**:

```
real_value  ≈  scale × int_value          e.g. 0.37 ≈ 0.0031 × 119
```

Now the datapath is `int8 × int8 → int32`. The scale factors never enter the inner loop; they are combined into **one** constant per layer that is applied once per output (`mult / 2^shift`, the "requantization" step). Detailed treatment in `03_software_stack.md` §2. What matters for architecture:

| Choice | Effect |
|---|---|
| int8 instead of fp32 | 4× less memory and bandwidth per number |
| int8 multiply + int32 add instead of fp32 multiply + fp32 add | ≈ **15×** less arithmetic energy (0.2 + 0.1 pJ vs 3.7 + 0.9 pJ, table below) |
| accuracy | here **96.94 % float → 96.94 % INT8** on the 360 test images (`sw/quantize.py`), and the RTL reproduces the INT8 result **bit-exactly** |

---

## 5. Why data movement, not arithmetic, is the problem

### 5.1 The energy table every accelerator designer knows

Approximate energy per operation in a 45 nm process (M. Horowitz, *"Computing's Energy Problem"*, ISSCC 2014; the numbers below are as tabulated in Hubara et al., arXiv:1609.07061). They are old and process-dependent, so trust the **ratios**, not the absolute values:

| Operation | Energy |
|---|---|
| 8-bit integer add | 0.03 pJ |
| 8-bit integer multiply | 0.2 pJ |
| 32-bit integer add | 0.1 pJ |
| 32-bit integer multiply | 3.1 pJ |
| 32-bit floating-point add | 0.9 pJ |
| 32-bit floating-point multiply | 3.7 pJ |
| Read 64 bits from an 8 KB SRAM | 10 pJ |
| Read 64 bits from a 32 KB SRAM | 20 pJ |
| Read 64 bits from a 1 MB SRAM | 100 pJ |
| Read 64 bits from DRAM | 1 300 – 2 600 pJ |

Read it as a hardware designer:

* One int8 MAC (0.2 pJ multiply + 0.1 pJ 32-bit accumulate) ≈ **0.3 pJ**.
* Fetching the **two operand bytes** of that MAC from an 8 KB SRAM ≈ 2 × 1.25 pJ ≈ **2.5 pJ** — about **8×** the MAC itself.
* Fetching them from DRAM: hundreds to thousands of times the MAC.

**So the question is never "how fast can I multiply?" It is "how many times do I have to fetch each operand?"** An accelerator is a machine for fetching each byte *once* and using it *many* times.

### 5.2 The three kinds of reuse in a matrix product

For `Y[M×N] = X[M×K] · W[K×N]` every element gets used many times, and each kind of reuse has a hardware trick:

| Operand | Used how many times | Hardware trick | In this project |
|---|---|---|---|
| a weight `W[k][n]` | `M` times (once per image in the batch) | **hold it in a register** next to the multiplier: *weight-stationary* | PE weight register |
| an input activation `X[m][k]` | `N` times (once per output neuron) | broadcast it along a row of MACs; keep a copy in a small buffer for the next tile | systolic array rows; Local Reuse Buffer (LRB) |
| a partial sum | `K` times (once per input) | pass it to the neighbour instead of writing it to memory | psum wires between PE rows; accumulator SRAM when `K` exceeds the array |

### 5.3 The roofline model — two speed limits

Your machine has **two** ceilings:

1. **Peak compute** — MACs per cycle if every PE is busy (ours: 4 × 4 = **16 MAC/cycle**).
2. **Memory bandwidth** — bytes per cycle you can deliver (ours: the weight SRAM gives one 32-bit word = **4 weights/cycle**).

The **arithmetic intensity** `I` is *how many MACs you get out of each byte fetched*. Attainable speed is

```
MAC/cycle  =  min( peak compute ,  bandwidth × I )
```

Worked example with our numbers. Take one fully-connected layer and look only at weights:

| Situation | Uses per weight byte (`I`) | Bandwidth-limited speed | Attainable |
|---|---|---|---|
| batch 1, no weight register | 1 | 4 B/cycle × 1 = **4 MAC/cycle** | 4 (25 % of peak) — *memory-bound* |
| batch 8, weight-stationary | 8 | 4 × 8 = 32 | min(16, 32) = **16** — *compute-bound* |

That one table is the reason for (a) the weight register in every PE and (b) processing a batch of images per weight load. Plotted, the roofline is a sloped line (memory-bound region) meeting a flat ceiling (compute-bound region); the "ridge point" is where `I × bandwidth = peak`.

---

## 6. Why plain CPUs and GPUs are inefficient at this (a fair account)

**CPU.** Each MAC is (at least) an instruction: fetch it, decode it, read two operands from the register file, execute, write back, and every operand load goes through the cache hierarchy. On a general-purpose core the arithmetic itself is a small fraction of the energy per instruction; most of it goes to *supporting* the instruction (this is the central point of Horowitz's ISSCC 2014 talk). SIMD units amortize the overhead over 8–64 lanes, but the operands still stream through registers and caches, and fp32 is 4× as many bytes as int8.

**GPU.** Thousands of MAC lanes and huge bandwidth to on-package DRAM make GPUs excellent — especially for training and large batches. Their inefficiency for *deployed inference* is different: every operation still reads its operands from a big general register file/shared memory, the machine must stay programmable for any workload (area and power spent on flexibility), power is in the hundreds of watts, and small-batch latency utilizes them poorly. Modern GPUs contain **tensor cores**, which are themselves small systolic-style matrix engines — the same idea this project implements. That is the honest picture: an accelerator is not doing different maths, it is doing the same maths in a machine whose *only* job is this pattern.

**Dedicated accelerator (systolic array).** Operands arrive from a **neighbouring register over a short wire**, not from a register file or cache. One weight fetch feeds `M` MACs; one activation fetch feeds a whole row; one partial sum hops to the next PE instead of visiting memory. No instruction stream at all — a small FSM sequences fixed tiles. (Google's first TPU, MIT's Eyeriss, Berkeley's Gemmini and NVIDIA's NVDLA are all in this family; they differ mostly in *which* operand stays put.)

| Design | Where a MAC gets its operands | Control overhead |
|---|---|---|
| CPU | register file ← cache ← DRAM | instruction fetch/decode per operation |
| GPU | big register file / shared memory | warp scheduler, per-thread instructions |
| Systolic array | neighbour PE register (one short wire) | one small FSM per tile |

---

## 7. So: does *this* project solve all of that? Is it more efficient?

The complete, honest, numbers-backed answer is in **`00_START_HERE.md` §2**. In four lines:

* The **first delivery** (v1) gave you a correct, parameterized INT8 systolic **matrix-multiply core** with a reuse buffer. On its own it could not run a neural network: no accumulation across input tiles (`K > ROWS`), no bias/requantize/ReLU, no layer chaining, no software flow.
* The **current delivery** (v2) adds all of that for fully-connected networks and proves it: a real PyTorch-trained network runs on the RTL and gives **bit-identical** answers to the integer reference (**96.94 %** on 360 test images).
* **Speed:** 5.35 useful MACs/cycle on a 16-MAC array (33 % utilization); 221 cycles per image at batch 8 — about **5×** faster than a 1-MAC/cycle datapath.
* **Data movement:** at the default 4×4 size, *partial sums* dominate and the design moves ≈ **2.25 bytes per MAC** on chip versus 2.0 for a design with no reuse at all — i.e. **not** better in bytes. A 16×16 array moves ≈ **0.43** (4.7× better). Energy in pJ is **not measured** (no synthesis has been run); `00_START_HERE.md` Step 7 and `06_verification_and_backend.md` §3.4 say how you get it.

---

## 8. Glossary (hardware translations)

| ML word | Hardware meaning |
|---|---|
| **Neuron** | one dot product + bias + squash; one column of `W` |
| **Layer** | `N` neurons sharing the same `K` inputs = one matrix product |
| **Fully-connected (FC / dense / Gemm)** | every input connects to every neuron — the layer type this chip runs |
| **Convolution** | the same small weight window slid over an image; can be *rewritten* as a matrix product (im2col) — not implemented in RTL here |
| **Weights / parameters** | learned constants |
| **Bias** | per-neuron constant added to the sum |
| **Activation (data)** | the values between layers |
| **Activation function / ReLU** | `max(0, x)` |
| **Logits / scores** | last layer's raw outputs; `argmax` picks the class |
| **Batch (M)** | how many inputs share one weight load |
| **Tensor / shape** | N-D array / its dimensions |
| **Inference / training** | evaluate frozen weights / find the weights |
| **MAC** | multiply-accumulate; 1 MAC = 2 "ops" in marketing TOPS |
| **GEMM** | general matrix-matrix multiply |
| **Quantization** | replacing floats by `scale × int` |
| **Scale / zero-point** | the conversion factor (and offset, in asymmetric schemes; we use symmetric = no offset) |
| **Calibration** | running representative data through the float network to pick output scales |
| **Requantization** | converting an int32 accumulator back to int8: `(acc · mult) >> shift` |
| **PTQ / QAT** | post-training quantization (what we do) / quantization-aware training (retrain with quantization in the loop) |
| **Tile / tiling** | cutting a big matrix into blocks that fit the array |
| **Weight-stationary / output-stationary** | which operand stays in the PE (weights / partial sums) |
| **Systolic array** | a grid of MACs where data marches neighbour-to-neighbour each clock |
| **Arithmetic intensity** | MACs per byte moved |
| **Roofline** | speed = min(peak compute, bandwidth × intensity) |
| **PPU** | our *post-processing unit*: bias + requantize + ReLU + saturate |
| **ONNX** | a file format that stores a network as a graph of operations plus weights |
| **Golden model** | a trusted software model whose outputs the RTL must match exactly |

---

## 9. Check yourself

1. **Shapes.** A layer has K = 64, N = 16, batch M = 8, on a 4×4 array. How many K-tiles, N-tiles and array passes? *(16, 4, 64.)* Where do you see these in the code? *(`sw/compile.py` prints them; `KT`, `NT` in the descriptors.)*
2. **Reuse.** In that layer, how many times is each weight used inside the array per pass? *(M = 8 times, once per image in the batch.)*
3. **Energy.** Why is a 5× speed-up in cycles not automatically a 5× energy saving? *(Energy follows bytes moved and bits toggled, not cycles; the byte count at 4×4 is barely better than no-reuse because of partial sums.)*
4. **Quantization.** What single hardware operation turns an int32 accumulator into the next layer's int8 input? *(`clamp(((acc + bias) · mult + round) >> shift)`, in `rtl/dnn_ppu.v`.)*

Next: **`02_project_mapping.md`** — what exactly you are building, block by block.
