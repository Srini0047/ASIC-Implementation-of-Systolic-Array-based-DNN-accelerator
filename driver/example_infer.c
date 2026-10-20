/*============================================================================
 * example_infer.c - run the whole INT8 network through the driver, with a
 * "recorder" HAL instead of real hardware.
 *
 * Why a recorder? This project has no chip and no FPGA yet. So instead of
 * poking a bus, the HAL writes every register access to a text file in the same
 * 16-hex-digit format the Verilog testbench (tb/tb_dnn_player.v) replays:
 *
 *      OP(2) 00 ADDR(4) DATA(8)      01=WRITE 03=READ_EXPECT 04=READ_LOG
 *                                    02=WAIT_DONE 05=WAIT_RD_VALID
 *
 * Two checks are then made by scripts/check_driver.sh:
 *   (a) this C trace is byte-identical to what the Python compiler generated
 *       (so the two independent implementations agree on the register protocol);
 *   (b) the C trace, replayed against the RTL, passes (so the RTL really does
 *       compute what the driver asked for).
 *
 * The recorder also plays "the device" for the few reads the driver performs:
 * ID/CFG registers come from the NET_HW_* macros in build/net_image.h and the
 * output words come from the expected result NET_Y (which, in the replay, the
 * RTL must reproduce).
 *
 *   build:  gcc -std=c99 -Wall -Wextra -Idriver -Ibuild driver/dnn_accel.c \
 *               driver/example_infer.c -o build/example_infer
 *   run:    ./build/example_infer build/c_trace.hex
 *==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnn_accel.h"
#include "net_image.h"      /* generated: NET[], NET_X[], NET_Y[], NET_HW_* */

/* ---- recorder HAL ------------------------------------------------------- */
typedef struct {
    FILE    *f;
    long     lines;
    uint32_t desc[DNN_MAX_LAYERS * 16];   /* shadow of the descriptor writes  */
    uint32_t n_layers;
    uint32_t rd_addr;                     /* last ACT_RD_ADDR written         */
} rec_t;

static void emit(rec_t *r, unsigned op, uint32_t addr, uint32_t data)
{
    fprintf(r->f, "%02x00%04x%08x\n", op, (unsigned)(addr & 0xFFFFu), (unsigned)data);
    r->lines++;
}

static void rec_write(void *ctx, uint16_t a, uint32_t v)
{
    rec_t *r = (rec_t *)ctx;
    emit(r, 0x01, a, v);
    if (a >= DNN_DESC_BASE && a < DNN_DESC_BASE + DNN_MAX_LAYERS * 16) r->desc[a - DNN_DESC_BASE] = v;
    if (a == DNN_REG_NLAYERS) r->n_layers = v;
    if (a == DNN_REG_ACT_RD_ADDR) r->rd_addr = v;
}

/* Expected activation word (as the 32-bit beat `beat`) for the last ACT_RD_ADDR. */
static uint32_t expected_beat(const rec_t *r, int beat)
{
    const int last = (int)r->n_layers - 1;
    const int dst = (int)r->desc[last * 16 + DNN_F_DST];
    const int nt_n = (int)r->desc[last * 16 + DNN_F_NT];
    const int idx = (int)r->rd_addr - dst;
    const int m = idx / nt_n, nt = idx % nt_n;
    const int np = NET[NET_LAYERS - 1].np;
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) {
        int lane = 4 * beat + j;
        if (lane < NET_HW_ROWS)
            v |= (uint32_t)(uint8_t)NET_Y[(size_t)m * np + nt * NET_HW_ROWS + lane] << (8 * j);
    }
    return v;
}

static uint32_t rec_read(void *ctx, uint16_t a)
{
    rec_t *r = (rec_t *)ctx;
    uint32_t v;
    switch (a) {
    case DNN_REG_ID:         v = DNN_ID_VALUE; break;
    case DNN_REG_CFG_ARRAY:  v = ((uint32_t)NET_HW_ROWS << 24) | ((uint32_t)NET_HW_M_MAX << 16) |
                                 ((uint32_t)NET_HW_KT_MAX << 8) | (uint32_t)NET_HW_NT_MAX; break;
    case DNN_REG_CFG_LAYERS: v = NET_HW_LAYERS; break;
    case DNN_REG_CFG_ACT:    v = NET_HW_ACT;    break;
    case DNN_REG_CFG_WGT:    v = NET_HW_WGT;    break;
    case DNN_REG_CFG_BIAS:   v = NET_HW_BIAS;   break;
    case DNN_REG_CFG_ACC:    v = NET_HW_ACC;    break;
    case DNN_REG_CFG_LRB:    v = NET_HW_LRB;    break;
    case DNN_REG_PERF_CYCLES: case DNN_REG_PERF_ACT_RD: case DNN_REG_PERF_LRB_RD:
    case DNN_REG_PERF_WGT_RD: case DNN_REG_PERF_VEC:
        emit(r, 0x04, a, 0);                 /* READ_LOG: value is measured by the RTL */
        return 0;
    default:
        if (a >= DNN_REG_ACT_RD_DATA && a < DNN_REG_ACT_RD_DATA + 8) {
            v = expected_beat(r, a - DNN_REG_ACT_RD_DATA);
            break;
        }
        fprintf(stderr, "recorder: unexpected read of register 0x%04x\n", a);
        exit(2);
    }
    emit(r, 0x03, a, v);                     /* READ_EXPECT: RTL must return exactly v */
    return v;
}

static int rec_wait(void *ctx, int kind)
{
    emit((rec_t *)ctx, kind == DNN_WAIT_DONE ? 0x02 : 0x05, 0, 0);
    return DNN_OK;
}

/* ---- main --------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "build/c_trace.hex";
    rec_t rec;
    memset(&rec, 0, sizeof rec);
    rec.f = fopen(out, "w");
    if (!rec.f) { perror(out); return 1; }

    dnn_hal_t hal = { &rec, rec_read, rec_write, rec_wait };
    dnn_dev_t dev;
    int rc;

    /* 1. who are you? */
    if ((rc = dnn_probe(&dev, &hal)) != DNN_OK) { fprintf(stderr, "probe failed: %d\n", rc); return 1; }
    printf("device: %dx%d array, M_MAX=%d KT_MAX=%d NT_MAX=%d, %d layers max\n",
           dev.info.rows, dev.info.cols, dev.info.m_max, dev.info.kt_max, dev.info.nt_max,
           dev.info.num_layers);

    /* 2. weights + biases + descriptors (once per network) */
    if ((rc = dnn_load_network(&dev, NET, NET_LAYERS, NET_BATCH)) != DNN_OK) {
        fprintf(stderr, "load failed: %d\n", rc); return 1;
    }
    /* 3. inputs (NET_X is already padded to layer-0 kp) */
    if ((rc = dnn_write_input(&dev, NET_X, NET[0].kp)) != DNN_OK) {
        fprintf(stderr, "input failed: %d\n", rc); return 1;
    }
    /* 4. go */
    if ((rc = dnn_run(&dev)) != DNN_OK) { fprintf(stderr, "run failed: %d\n", rc); return 1; }

    dnn_perf_t perf;
    dnn_read_perf(&dev, &perf);

    /* 5. results */
    const int np = NET[NET_LAYERS - 1].np;
    int8_t y[NET_BATCH * 64];
    if (np * NET_BATCH > (int)sizeof y) { fprintf(stderr, "output buffer too small\n"); return 1; }
    if ((rc = dnn_read_output(&dev, y)) != DNN_OK) { fprintf(stderr, "read failed: %d\n", rc); return 1; }

    int bad = 0;
    for (int m = 0; m < NET_BATCH; m++) {
        printf("input %d -> class %d\n", m, dnn_argmax(&y[m * np], 10 < np ? 10 : np));
        for (int n = 0; n < np; n++)
            if (y[m * np + n] != NET_Y[m * np + n]) bad++;
    }
    fclose(rec.f);
    printf("trace: %ld register accesses -> %s\n", rec.lines, out);
    printf(bad ? "MISMATCH in %d output values\n" : "output unpacking OK\n", bad);
    return bad ? 1 : 0;
}
