/*============================================================================
 * dnn_accel.c - host driver for the systolic-array DNN accelerator.
 *
 * Read this file top to bottom: it is the SOFTWARE HALF of the hardware/software
 * contract, in the same order the hardware needs things:
 *
 *   dnn_probe         who are you and how big are you?
 *   dnn_load_network  put weights/biases in the SRAMs, write layer descriptors
 *   dnn_write_input   put the input vectors in the activation SRAM
 *   dnn_run           press START, wait for done
 *   dnn_read_output   read the last layer's results
 *
 * Every function is a sequence of hal->write()/hal->read() calls. The order and
 * addresses are identical to sw/compiler_lib.py (checked byte-for-byte by
 * scripts/check_driver.sh).
 *==========================================================================*/
#include "dnn_accel.h"

#include <string.h>

/* ---- tiny helpers ------------------------------------------------------- */
static uint32_t rd(const dnn_dev_t *d, uint32_t a) { return d->hal.read(d->hal.ctx, (uint16_t)a); }
static void     wr(const dnn_dev_t *d, uint32_t a, uint32_t v) { d->hal.write(d->hal.ctx, (uint16_t)a, v); }

static int pow2_ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }

/* Pack up to 4 signed 8-bit lanes (starting at lane `first`) into one 32-bit
 * register beat, lane 0 in bits [7:0].  Lanes >= `count` are zero. */
static uint32_t pack4(const int8_t *lanes, int first, int count)
{
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) {
        int idx = first + j;
        if (idx < count) v |= (uint32_t)(uint8_t)lanes[idx] << (8 * j);
    }
    return v;
}

/* ---- waiting ------------------------------------------------------------ */
static int wait_for(dnn_dev_t *d, int kind)
{
    if (d->hal.wait) return d->hal.wait(d->hal.ctx, kind);
    for (uint32_t i = 0; i < DNN_POLL_LIMIT; i++) {
        uint32_t st = rd(d, DNN_REG_STATUS);
        if (kind == DNN_WAIT_DONE) {
            if (st & DNN_ST_BUSY) continue;             /* err/done are only meaningful once !busy */
            if (st & DNN_ST_ERR) return DNN_E_HW;
            if (st & DNN_ST_DONE) return DNN_OK;
        } else if (st & DNN_ST_RD_VALID) {
            return DNN_OK;
        }
    }
    return DNN_E_TIMEOUT;
}

/* ---- 1. probe ----------------------------------------------------------- */
int dnn_probe(dnn_dev_t *d, const dnn_hal_t *hal)
{
    if (!d || !hal || !hal->read || !hal->write) return DNN_E_ARG;
    memset(d, 0, sizeof *d);
    d->hal = *hal;
    dnn_info_t *h = &d->info;

    h->id = rd(d, DNN_REG_ID);
    if (h->id != DNN_ID_VALUE) return DNN_E_ID;

    uint32_t a = rd(d, DNN_REG_CFG_ARRAY);
    h->rows       = (int)(a >> 24);
    h->cols       = h->rows;                    /* RTL requires ROWS == COLS */
    h->m_max      = (int)((a >> 16) & 0xFF);
    h->kt_max     = (int)((a >> 8) & 0xFF);
    h->nt_max     = (int)(a & 0xFF);
    h->num_layers = (int)rd(d, DNN_REG_CFG_LAYERS);
    h->act_depth  = (int)rd(d, DNN_REG_CFG_ACT);
    h->wgt_depth  = (int)rd(d, DNN_REG_CFG_WGT);
    h->bias_depth = (int)rd(d, DNN_REG_CFG_BIAS);
    h->acc_depth  = (int)rd(d, DNN_REG_CFG_ACC);
    h->lrb_depth  = (int)rd(d, DNN_REG_CFG_LRB);

    if (h->rows < 1 || h->m_max < 1 || h->kt_max < 1 || h->nt_max < 1) return DNN_E_ID;

    /* SRAM words are wider than the 32-bit bus: they are written in "beats". */
    h->beats_w  = (h->cols * 8 + 31) / 32;
    h->beats_a  = (h->rows * 8 + 31) / 32;
    h->beats_b  = h->cols;                      /* one INT32 bias lane per beat */
    h->stride_w = pow2_ceil(h->beats_w);
    h->stride_a = pow2_ceil(h->beats_a);
    h->stride_b = pow2_ceil(h->beats_b);
    h->bank     = h->m_max * h->kt_max;

    d->probed = 1;
    return DNN_OK;
}

/* ---- 2. load the network ----------------------------------------------- */
int dnn_load_network(dnn_dev_t *d, const dnn_layer_t *L, int n, int batch)
{
    if (!d || !d->probed || !L) return DNN_E_ARG;
    const dnn_info_t *h = &d->info;
    const int R = h->rows;

    /* ---- plan (pure software; nothing is written until the whole network fits) */
    if (n < 1 || n > h->num_layers || n > DNN_MAX_LAYERS) return DNN_E_CAPACITY;
    if (batch < 1 || batch > h->m_max) return DNN_E_CAPACITY;
    if (2 * h->bank > h->act_depth) return DNN_E_CAPACITY;

    int wptr = 0, bptr = 0;
    for (int i = 0; i < n; i++) {
        if (!L[i].w || !L[i].b) return DNN_E_ARG;
        if (L[i].kp < R || L[i].np < R || L[i].kp % R || L[i].np % R) return DNN_E_ARG;
        int kt = L[i].kp / R, nt = L[i].np / R;
        if (kt > h->kt_max || nt > h->nt_max) return DNN_E_CAPACITY;
        if (batch * kt > h->bank || batch * nt > h->bank) return DNN_E_CAPACITY;
        if (batch * nt > h->acc_depth) return DNN_E_CAPACITY;
        if (L[i].shift > 47u || L[i].mult >= 65536u) return DNN_E_CAPACITY;
        if (i > 0 && L[i - 1].np != L[i].kp) return DNN_E_ARG;

        d->lay[i].kp = L[i].kp;  d->lay[i].np = L[i].np;
        d->lay[i].kt = kt;       d->lay[i].nt = nt;
        d->lay[i].src = (i % 2 == 0) ? 0 : h->bank;     /* ping-pong regions */
        d->lay[i].dst = (i % 2 == 0) ? h->bank : 0;
        d->lay[i].wbase = wptr;  d->lay[i].bbase = bptr;
        wptr += kt * nt * R;
        bptr += nt;
    }
    if (wptr > h->wgt_depth || bptr > h->bias_depth) return DNN_E_CAPACITY;

    /* ---- program the hardware */
    for (int i = 0; i < n; i++) {
        const int kt_n = d->lay[i].kt, nt_n = d->lay[i].nt, np = L[i].np;

        /* Weight tiles in PASS ORDER p = kt*NT + nt (the order the controller
         * consumes them): each tile is ROWS words, word r = row r of the tile,
         * COLS int8 lanes wide. */
        int p = 0;
        for (int kt = 0; kt < kt_n; kt++) {
            for (int nt = 0; nt < nt_n; nt++, p++) {
                for (int r = 0; r < R; r++) {
                    const int8_t *row = &L[i].w[(size_t)(kt * R + r) * np + nt * R];
                    int word = d->lay[i].wbase + p * R + r;
                    for (int b = 0; b < h->beats_w; b++)
                        wr(d, DNN_WGT_BASE + (uint32_t)(word * h->stride_w + b), pack4(row, 4 * b, R));
                }
            }
        }
        /* Bias: one 32-bit lane per beat, one SRAM word per N-tile. */
        for (int nt = 0; nt < nt_n; nt++) {
            int word = d->lay[i].bbase + nt;
            for (int c = 0; c < R; c++)
                wr(d, DNN_BIAS_BASE + (uint32_t)(word * h->stride_b + c), (uint32_t)L[i].b[nt * R + c]);
        }
        /* Layer descriptor (field order = dnn_ctrl.v). */
        const uint32_t base = DNN_DESC_BASE + (uint32_t)(i * 16);
        wr(d, base + DNN_F_M,     (uint32_t)batch);
        wr(d, base + DNN_F_KT,    (uint32_t)kt_n);
        wr(d, base + DNN_F_NT,    (uint32_t)nt_n);
        wr(d, base + DNN_F_WBASE, (uint32_t)d->lay[i].wbase);
        wr(d, base + DNN_F_BBASE, (uint32_t)d->lay[i].bbase);
        wr(d, base + DNN_F_SRC,   (uint32_t)d->lay[i].src);
        wr(d, base + DNN_F_DST,   (uint32_t)d->lay[i].dst);
        wr(d, base + DNN_F_MULT,  L[i].mult);
        wr(d, base + DNN_F_SHIFT, L[i].shift);
        wr(d, base + DNN_F_RELU,  L[i].relu ? 1u : 0u);
    }
    wr(d, DNN_REG_NLAYERS, (uint32_t)n);

    d->n_layers = n;
    d->batch = batch;
    d->loaded = 1;
    return DNN_OK;
}

/* ---- 3. input ----------------------------------------------------------- */
int dnn_write_input(dnn_dev_t *d, const int8_t *x, int k_in)
{
    if (!d || !d->loaded || !x) return DNN_E_ARG;
    const dnn_info_t *h = &d->info;
    const int R = h->rows, kp0 = d->lay[0].kp, kt0 = d->lay[0].kt;
    if (k_in < 1 || k_in > kp0) return DNN_E_ARG;

    int8_t lanes[64];                                   /* one activation word */
    if (R > 64) return DNN_E_CAPACITY;
    for (int m = 0; m < d->batch; m++) {
        for (int kt = 0; kt < kt0; kt++) {
            for (int c = 0; c < R; c++) {               /* zero-pad K up to kp0 */
                int k = kt * R + c;
                lanes[c] = (k < k_in) ? x[(size_t)m * k_in + k] : 0;
            }
            int word = d->lay[0].src + m * kt0 + kt;
            for (int b = 0; b < h->beats_a; b++)
                wr(d, DNN_ACT_BASE + (uint32_t)(word * h->stride_a + b), pack4(lanes, 4 * b, R));
        }
    }
    return DNN_OK;
}

/* ---- 4. run ------------------------------------------------------------- */
int dnn_run(dnn_dev_t *d)
{
    if (!d || !d->loaded) return DNN_E_ARG;
    wr(d, DNN_REG_CTRL, 1u);                            /* START pulse */
    return wait_for(d, DNN_WAIT_DONE);
}

/* ---- 5. performance counters ------------------------------------------- */
int dnn_read_perf(dnn_dev_t *d, dnn_perf_t *p)
{
    if (!d || !d->probed || !p) return DNN_E_ARG;
    p->cycles = rd(d, DNN_REG_PERF_CYCLES);
    p->act_rd = rd(d, DNN_REG_PERF_ACT_RD);
    p->lrb_rd = rd(d, DNN_REG_PERF_LRB_RD);
    p->wgt_rd = rd(d, DNN_REG_PERF_WGT_RD);
    p->vec    = rd(d, DNN_REG_PERF_VEC);
    return DNN_OK;
}

/* ---- 6. output ---------------------------------------------------------- */
int dnn_read_output(dnn_dev_t *d, int8_t *y)
{
    if (!d || !d->loaded || !y) return DNN_E_ARG;
    const dnn_info_t *h = &d->info;
    const int R = h->rows, last = d->n_layers - 1;
    const int nt_n = d->lay[last].nt, np = d->lay[last].np, dst = d->lay[last].dst;

    for (int m = 0; m < d->batch; m++) {
        for (int nt = 0; nt < nt_n; nt++) {
            wr(d, DNN_REG_ACT_RD_ADDR, (uint32_t)(dst + m * nt_n + nt));
            int rc = wait_for(d, DNN_WAIT_RD_VALID);    /* SRAM read latency */
            if (rc != DNN_OK) return rc;
            for (int b = 0; b < h->beats_a; b++) {
                uint32_t v = rd(d, DNN_REG_ACT_RD_DATA + (uint32_t)b);
                for (int j = 0; j < 4; j++) {
                    int lane = 4 * b + j;
                    if (lane >= R) break;
                    int s = (int)((v >> (8 * j)) & 0xFFu);
                    y[(size_t)m * np + nt * R + lane] = (int8_t)(s >= 128 ? s - 256 : s);
                }
            }
        }
    }
    return DNN_OK;
}

int dnn_argmax(const int8_t *row, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (row[i] > row[best]) best = i;
    return best;
}
