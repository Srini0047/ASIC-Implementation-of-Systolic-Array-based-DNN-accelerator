// Lint waivers (intentional): narrow values are zero-extended into wider counters/addresses,
// and 16-bit descriptor / bus fields are truncated to the widths the hardware needs.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
//=============================================================================
// Module      : dnn_accel_top
// Description : Chip-level wrapper for the DNN inference accelerator:
//
//        CPU / AXI-Lite bridge / testbench
//                     |  32-bit register bus (see dnn_reg_if.v for the map)
//                     v
//   rst_n_i --> rst_sync --> dnn_reg_if --> dnn_accel_core
//                                             (dnn_ctrl, memories, LRB,
//                                              systolic array, accumulate, PPU)
//
//   Pins: clk, rst_n_i, reg_wr_en, reg_rd_en, reg_addr[15:0], reg_wdata[31:0],
//   reg_rdata[31:0], irq. About 85 signals - a realistic block-level pin count
//   compared with the ~350 of systolic_accel_top (matmul-only).
//
//   irq : level, high while `done` or `err` is set (cleared by the next START).
//
//   Parameters ending in _W / _AW are derived - do not override. ROWS must
//   equal COLS (output tile == next layer's input slice).
//=============================================================================
module dnn_accel_top #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter ACC_WIDTH  = 32,
    parameter ACT_DEPTH  = 256,
    parameter WGT_DEPTH  = 512,
    parameter BIAS_DEPTH = 16,
    parameter ACC_DEPTH  = 32,
    parameter LRB_DEPTH  = 8,
    parameter NUM_LAYERS = 4,
    parameter M_MAX      = 8,
    parameter KT_MAX     = 16,
    parameter NT_MAX     = 8,
    parameter PERF_EN    = 1,
    parameter SYNC_RST_IN = 1,

    // ---- derived (do not override) ----
    parameter ACT_AW  = (ACT_DEPTH  > 1) ? $clog2(ACT_DEPTH)  : 1,
    parameter WGT_AW  = (WGT_DEPTH  > 1) ? $clog2(WGT_DEPTH)  : 1,
    parameter BIAS_AW = (BIAS_DEPTH > 1) ? $clog2(BIAS_DEPTH) : 1,
    parameter LAY_W   = $clog2(NUM_LAYERS + 1),
    parameter DESC_AW = $clog2(NUM_LAYERS * 16)
)(
    input  wire        clk,
    input  wire        rst_n_i,

    input  wire        reg_wr_en,
    input  wire        reg_rd_en,
    input  wire [15:0] reg_addr,
    input  wire [31:0] reg_wdata,
    output wire [31:0] reg_rdata,

    output wire        irq
);

    wire rst_n;
    generate
        if (SYNC_RST_IN != 0) begin : g_rs
            rst_sync #(.STAGES(2)) u_rst_sync (.clk(clk), .arst_n(rst_n_i), .srst_n(rst_n));
        end else begin : g_nors
            assign rst_n = rst_n_i;
        end
    endgenerate

    wire                     start, busy, done, err;
    wire [LAY_W-1:0]         n_layers;
    wire                     desc_we, wgt_we, bias_we, act_we, act_re;
    wire [DESC_AW-1:0]       desc_waddr;
    wire [15:0]              desc_wdata;
    wire [WGT_AW-1:0]        wgt_waddr;
    wire [COLS*8-1:0]        wgt_wdata;
    wire [BIAS_AW-1:0]       bias_waddr;
    wire [COLS*ACC_WIDTH-1:0] bias_wdata;
    wire [ACT_AW-1:0]        act_waddr, act_raddr;
    wire [ROWS*8-1:0]        act_wdata, act_rdata;
    wire [31:0]              p_cyc, p_act, p_lrb, p_wgt, p_vec;

    dnn_reg_if #(
        .ROWS(ROWS), .COLS(COLS), .ACC_WIDTH(ACC_WIDTH), .ACT_DEPTH(ACT_DEPTH),
        .WGT_DEPTH(WGT_DEPTH), .BIAS_DEPTH(BIAS_DEPTH), .ACC_DEPTH(ACC_DEPTH),
        .LRB_DEPTH(LRB_DEPTH), .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX),
        .KT_MAX(KT_MAX), .NT_MAX(NT_MAX), .ACT_AW(ACT_AW), .WGT_AW(WGT_AW),
        .BIAS_AW(BIAS_AW), .LAY_W(LAY_W), .DESC_AW(DESC_AW)
    ) u_if (
        .clk(clk), .rst_n(rst_n),
        .reg_wr_en(reg_wr_en), .reg_rd_en(reg_rd_en), .reg_addr(reg_addr),
        .reg_wdata(reg_wdata), .reg_rdata(reg_rdata),
        .start(start), .n_layers(n_layers), .busy(busy), .done(done), .err(err),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .wgt_we(wgt_we), .wgt_waddr(wgt_waddr), .wgt_wdata(wgt_wdata),
        .bias_we(bias_we), .bias_waddr(bias_waddr), .bias_wdata(bias_wdata),
        .act_we(act_we), .act_waddr(act_waddr), .act_wdata(act_wdata),
        .act_re(act_re), .act_raddr(act_raddr), .act_rdata(act_rdata),
        .perf_cycles(p_cyc), .perf_act_rd(p_act), .perf_lrb_rd(p_lrb),
        .perf_wgt_rd(p_wgt), .perf_vec(p_vec));

    dnn_accel_core #(
        .ROWS(ROWS), .COLS(COLS), .ACC_WIDTH(ACC_WIDTH), .ACT_DEPTH(ACT_DEPTH),
        .WGT_DEPTH(WGT_DEPTH), .BIAS_DEPTH(BIAS_DEPTH), .ACC_DEPTH(ACC_DEPTH),
        .LRB_DEPTH(LRB_DEPTH), .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX),
        .KT_MAX(KT_MAX), .NT_MAX(NT_MAX), .PERF_EN(PERF_EN)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(start), .n_layers(n_layers), .busy(busy), .done(done), .err(err),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .wgt_we(wgt_we), .wgt_waddr(wgt_waddr), .wgt_wdata(wgt_wdata),
        .bias_we(bias_we), .bias_waddr(bias_waddr), .bias_wdata(bias_wdata),
        .act_we(act_we), .act_waddr(act_waddr), .act_wdata(act_wdata),
        .act_re(act_re), .act_raddr(act_raddr), .act_rdata(act_rdata),
        .perf_cycles(p_cyc), .perf_act_rd(p_act), .perf_lrb_rd(p_lrb),
        .perf_wgt_rd(p_wgt), .perf_vec(p_vec));

    assign irq = done | err;

endmodule
