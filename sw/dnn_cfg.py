"""Shared hardware description used by the compiler, golden model and reports.

Everything here mirrors rtl/dnn_reg_if.v and the parameters of
rtl/dnn_accel_top.v. If you change a Verilog parameter, change it here too
(or pass the same value on the command line of sw/compile.py).
"""
from dataclasses import dataclass


def clog2(x: int) -> int:
    return 0 if x <= 1 else (x - 1).bit_length()


# ---- register map (32-bit word addresses) ---------------------------------
REG_CTRL, REG_STATUS, REG_NLAYERS, REG_ACT_RD_ADDR = 0x0000, 0x0001, 0x0002, 0x0003
REG_ID = 0x0004
REG_CFG_ARRAY, REG_CFG_LAYERS = 0x0005, 0x0006
REG_CFG_ACT, REG_CFG_WGT, REG_CFG_BIAS, REG_CFG_ACC, REG_CFG_LRB = 0x0007, 0x0008, 0x0009, 0x000A, 0x000B
REG_ACT_RD_DATA = 0x0010            # + beat
REG_PERF_CYCLES, REG_PERF_ACT_RD, REG_PERF_LRB_RD = 0x0020, 0x0021, 0x0022
REG_PERF_WGT_RD, REG_PERF_VEC = 0x0023, 0x0024
DESC_BASE, WGT_BASE, ACT_BASE, BIAS_BASE = 0x0100, 0x1000, 0x2000, 0x3000
ID_VALUE = 0xD44A0100

STATUS_BUSY, STATUS_DONE, STATUS_ERR, STATUS_RD_VALID = 1, 2, 4, 8

# descriptor field indices (must match dnn_ctrl.v)
F_M, F_KT, F_NT, F_WBASE, F_BBASE, F_SRC, F_DST, F_MULT, F_SHIFT, F_RELU = range(10)

# ---- program opcodes for the testbench player -----------------------------
OP_WRITE, OP_WAIT_DONE, OP_READ_EXPECT, OP_READ_LOG = 0x01, 0x02, 0x03, 0x04
OP_WAIT_RD_VALID, OP_DELAY, OP_WAIT_DONE_ERR, OP_END = 0x05, 0x06, 0x07, 0xFF


@dataclass
class HwCfg:
    """Hardware configuration; defaults equal the RTL defaults."""
    rows: int = 4                # = cols
    acc_width: int = 32
    act_depth: int = 256
    wgt_depth: int = 512
    bias_depth: int = 16
    acc_depth: int = 32
    lrb_depth: int = 8
    num_layers: int = 4
    m_max: int = 8
    kt_max: int = 16
    nt_max: int = 8

    @property
    def cols(self): return self.rows
    @property
    def beats_w(self): return (self.cols * 8 + 31) // 32
    @property
    def beats_a(self): return (self.rows * 8 + 31) // 32
    @property
    def beats_b(self): return self.cols
    @property
    def stride_w(self): return 1 << clog2(self.beats_w)
    @property
    def stride_a(self): return 1 << clog2(self.beats_a)
    @property
    def stride_b(self): return 1 << clog2(self.beats_b)
    @property
    def bank(self):            # size of one ping-pong region of the act SRAM
        return self.m_max * self.kt_max


def pack_lanes(vals, bits):
    """Little-endian pack of signed ints into one unsigned word."""
    w = 0
    for i, v in enumerate(vals):
        w |= (int(v) & ((1 << bits) - 1)) << (bits * i)
    return w


def split_beats(word, nbeats):
    return [(word >> (32 * b)) & 0xFFFFFFFF for b in range(nbeats)]
