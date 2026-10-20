/*============================================================================
 * dnn_accel.h - host driver API for the systolic-array DNN accelerator.
 *
 * WHAT THIS IS
 *   The accelerator (rtl/dnn_accel_top.v) exposes ONE 32-bit register bus.
 *   Everything - loading weights, describing layers, pushing an input, pressing
 *   START, reading results - is a sequence of 32-bit register reads/writes.
 *   This driver is that sequence, written in portable C99.
 *
 * HOW IT TALKS TO HARDWARE
 *   The driver never touches memory-mapped addresses directly. You give it a
 *   tiny "HAL" (hardware abstraction layer) = 2 or 3 function pointers:
 *       read (ctx, word_addr)          -> 32-bit value
 *       write(ctx, word_addr, value)
 *       wait (ctx, kind)   [optional]  -> block until done / rd_valid
 *   * On a real SoC the HAL is a volatile pointer into the AXI-Lite window
 *     (see hal_mmio.h - 10 lines).
 *   * In this project the HAL is a "recorder" (driver/example_infer.c) that
 *     writes every access to a file which the RTL testbench then replays.
 *
 * UNITS: addresses are 32-bit WORD addresses (the RTL reg_addr bus), not byte
 * addresses. A byte-addressed AXI-Lite bridge multiplies by 4 (hal_mmio.h does).
 *
 * MUST STAY IN SYNC WITH: rtl/dnn_reg_if.v, rtl/dnn_ctrl.v, sw/dnn_cfg.py,
 *   sw/compiler_lib.py. scripts/check_driver.sh proves the C register stream is
 *   byte-identical to the Python one.
 *==========================================================================*/
#ifndef DNN_ACCEL_H
#define DNN_ACCEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- register map (word addresses) ------------------------------------- */
#define DNN_REG_CTRL          0x0000u  /* bit0 = START (self-clearing pulse)  */
#define DNN_REG_STATUS        0x0001u  /* see DNN_ST_*                        */
#define DNN_REG_NLAYERS       0x0002u  /* number of layers to run (1..NUM_LAYERS) */
#define DNN_REG_ACT_RD_ADDR   0x0003u  /* activation word to read back        */
#define DNN_REG_ID            0x0004u  /* must read DNN_ID_VALUE              */
#define DNN_REG_CFG_ARRAY     0x0005u  /* {ROWS[31:24], M_MAX, KT_MAX, NT_MAX} */
#define DNN_REG_CFG_LAYERS    0x0006u
#define DNN_REG_CFG_ACT       0x0007u
#define DNN_REG_CFG_WGT       0x0008u
#define DNN_REG_CFG_BIAS      0x0009u
#define DNN_REG_CFG_ACC       0x000Au
#define DNN_REG_CFG_LRB       0x000Bu
#define DNN_REG_ACT_RD_DATA   0x0010u  /* + beat index                        */
#define DNN_REG_PERF_CYCLES   0x0020u
#define DNN_REG_PERF_ACT_RD   0x0021u
#define DNN_REG_PERF_LRB_RD   0x0022u
#define DNN_REG_PERF_WGT_RD   0x0023u
#define DNN_REG_PERF_VEC      0x0024u
#define DNN_DESC_BASE         0x0100u  /* + layer*16 + field                  */
#define DNN_WGT_BASE          0x1000u  /* weight-SRAM window                  */
#define DNN_ACT_BASE          0x2000u  /* activation-SRAM window              */
#define DNN_BIAS_BASE         0x3000u  /* bias-SRAM window                    */

#define DNN_ID_VALUE          0xD44A0100u

/* STATUS bits */
#define DNN_ST_BUSY           0x1u
#define DNN_ST_DONE           0x2u
#define DNN_ST_ERR            0x4u
#define DNN_ST_RD_VALID       0x8u

/* descriptor field indices (dnn_ctrl.v) */
enum {
    DNN_F_M = 0, DNN_F_KT, DNN_F_NT, DNN_F_WBASE, DNN_F_BBASE,
    DNN_F_SRC, DNN_F_DST, DNN_F_MULT, DNN_F_SHIFT, DNN_F_RELU
};

/* wait kinds passed to hal->wait() */
#define DNN_WAIT_DONE         0
#define DNN_WAIT_RD_VALID     1

/* return codes */
#define DNN_OK                0
#define DNN_E_ARG            (-1)  /* bad argument / call order            */
#define DNN_E_ID             (-2)  /* ID register mismatch: wrong device   */
#define DNN_E_CAPACITY       (-3)  /* network does not fit this hardware   */
#define DNN_E_HW             (-4)  /* hardware raised STATUS.err           */
#define DNN_E_TIMEOUT        (-5)  /* polling gave up                      */

#define DNN_MAX_LAYERS        16
#define DNN_POLL_LIMIT        1000000u

/* ---- hardware abstraction layer ---------------------------------------- */
typedef struct {
    void     *ctx;
    uint32_t (*read)(void *ctx, uint16_t word_addr);
    void     (*write)(void *ctx, uint16_t word_addr, uint32_t data);
    /* Optional. NULL => the driver polls DNN_REG_STATUS itself.
     * Provide it to sleep on the interrupt line instead of polling.
     * Return DNN_OK, DNN_E_HW or DNN_E_TIMEOUT. */
    int      (*wait)(void *ctx, int kind);
} dnn_hal_t;

/* ---- what the hardware says it is (read from the CFG_* registers) ------- */
typedef struct {
    uint32_t id;
    int rows, cols;                         /* systolic array size (rows == cols) */
    int m_max, kt_max, nt_max, num_layers;
    int act_depth, wgt_depth, bias_depth, acc_depth, lrb_depth;
    int beats_w, beats_a, beats_b;          /* 32-bit register beats per SRAM word */
    int stride_w, stride_a, stride_b;       /* beats rounded up to a power of two   */
    int bank;                               /* words in one ping-pong region        */
} dnn_info_t;

/* ---- one fully-connected layer, as produced by sw/compile.py ------------
 *   y = clamp( ((x*W + bias) * mult + round) >> shift )   (ReLU folded into clamp)
 * kp, np are the PADDED dimensions (multiples of rows). w is [kp][np] row-major
 * int8, bias is [np] int32. The Python compiler already zero-padded them. */
typedef struct {
    int kp, np, relu;
    uint32_t mult, shift;
    const int8_t  *w;
    const int32_t *b;
} dnn_layer_t;

typedef struct {
    uint32_t cycles, act_rd, lrb_rd, wgt_rd, vec;
} dnn_perf_t;

/* ---- driver state (opaque to the application) --------------------------- */
typedef struct {
    dnn_hal_t  hal;
    dnn_info_t info;
    int        probed, loaded;
    int        n_layers, batch;
    struct { int kt, nt, src, dst, kp, np, wbase, bbase; } lay[DNN_MAX_LAYERS];
} dnn_dev_t;

/* ---- API ---------------------------------------------------------------- */
/* 1. Discover the hardware. Reads ID + the 7 CFG registers. */
int dnn_probe(dnn_dev_t *d, const dnn_hal_t *hal);

/* 2. Program weights, biases and layer descriptors (do this once per network).
 *    `batch` = number of input vectors processed per run (1..info.m_max). */
int dnn_load_network(dnn_dev_t *d, const dnn_layer_t *layers, int n_layers, int batch);

/* 3. Write a batch of inputs: x is [batch][k_in] int8 (k_in <= layer-0 kp;
 *    the rest is zero-padded exactly as the compiler pads). */
int dnn_write_input(dnn_dev_t *d, const int8_t *x, int k_in);

/* 4. Press START and wait for done. Returns DNN_E_HW if the hardware rejected
 *    the descriptors (STATUS.err). */
int dnn_run(dnn_dev_t *d);

/* 5. Optional performance counters (valid after dnn_run). */
int dnn_read_perf(dnn_dev_t *d, dnn_perf_t *p);

/* 6. Read the last layer's output: y is [batch][np_last] int8. */
int dnn_read_output(dnn_dev_t *d, int8_t *y);

/* helper: largest index of y[m][:] (classification). */
int dnn_argmax(const int8_t *row, int n);

#ifdef __cplusplus
}
#endif
#endif /* DNN_ACCEL_H */
