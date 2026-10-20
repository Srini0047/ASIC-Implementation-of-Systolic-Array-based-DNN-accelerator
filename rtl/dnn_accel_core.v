// Lint waivers (intentional): narrow values are zero-extended into wider counters/addresses,
// and 16-bit descriptor / bus fields are truncated to the widths the hardware needs.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
//=============================================================================
// Module      : dnn_accel_core
// Description : Multi-layer INT8 fully-connected inference engine built around
//               the verified systolic core (sa_core) and LRB.
//
//   Adds, relative to systolic_accel_top (matmul only):
//     1. K-tile ACCUMULATION  : partial sums of successive K slices are added
//        in an accumulator SRAM (read-modify-write, one word per cycle).
//     2. PPU                  : bias + requantize + ReLU/clamp -> INT8.
//     3. Ping-pong write-back : the PPU writes the layer output straight into
//        the activation SRAM, in the layout the next layer reads. Two regions
//        (SRC/DST alternate) = ping-pong buffers, on a single 1R1W SRAM.
//     4. Layer sequencer      : dnn_ctrl runs all layers from descriptors.
//
//   Data path per pass (one weight tile, M activation vectors):
//
//     act SRAM --+--> LRB/feed --> sa_core --> res_vec --+
//                |                                       v
//                |   acc SRAM --(read, 1 cycle early)--> [+] --> s1 regs
//                |       ^                                        |   |
//                |       +---------- write (not last K) ----------+   |
//                |                                                    v
//     bias SRAM ------------------------------------------------>  dnn_ppu
//                ^                                                    |
//                +----------------- write INT8 (last K) --------------+
//
//   Timing of the read-modify-write: sa_core raises res_vld_early one cycle
//   before res_vld, so the accumulator read issued then returns its data in
//   the same cycle as res_vec.
//
//   Host memory ports are only honoured while !busy. Requires ROWS == COLS so
//   an output tile (COLS INT8) is exactly one K-slice (ROWS INT8) of the next
//   layer.
//=============================================================================
module dnn_accel_core #(
    parameter ROWS       = 4,
    parameter COLS       = 4,        // must equal ROWS
    parameter ACC_WIDTH  = 32,
    parameter ACT_DEPTH  = 256,      // words of ROWS x INT8 (two ping-pong regions)
    parameter WGT_DEPTH  = 512,      // words of COLS x INT8
    parameter BIAS_DEPTH = 16,       // words of COLS x INT32
    parameter ACC_DEPTH  = 32,       // words of COLS x INT32 (>= M*NT)
    parameter LRB_DEPTH  = 8,
    parameter NUM_LAYERS = 4,
    parameter M_MAX      = 8,        // max batch rows per layer
    parameter KT_MAX     = 16,       // max K tiles
    parameter NT_MAX     = 8,        // max N tiles
    parameter PERF_EN    = 1,

    // ---- derived widths (do not override) ----
    parameter M_W     = $clog2(M_MAX + 1),
    parameter KT_W    = $clog2(KT_MAX + 1),
    parameter NT_W    = $clog2(NT_MAX + 1),
    parameter ACT_AW  = (ACT_DEPTH  > 1) ? $clog2(ACT_DEPTH)  : 1,
    parameter WGT_AW  = (WGT_DEPTH  > 1) ? $clog2(WGT_DEPTH)  : 1,
    parameter BIAS_AW = (BIAS_DEPTH > 1) ? $clog2(BIAS_DEPTH) : 1,
    parameter ACC_AW  = (ACC_DEPTH  > 1) ? $clog2(ACC_DEPTH)  : 1,
    parameter LRB_AW  = (LRB_DEPTH  > 1) ? $clog2(LRB_DEPTH)  : 1,
    parameter LAY_W   = $clog2(NUM_LAYERS + 1),
    parameter DESC_AW = $clog2(NUM_LAYERS * 16)
)(
    input  wire                     clk,
    input  wire                     rst_n,          // sync, active low

    // command / status
    input  wire                     start,
    input  wire [LAY_W-1:0]         n_layers,
    output wire                     busy,
    output wire                     done,
    output wire                     err,

    // descriptor write
    input  wire                     desc_we,
    input  wire [DESC_AW-1:0]       desc_waddr,
    input  wire [15:0]              desc_wdata,

    // weight / bias / activation memory write ports (host, idle only)
    input  wire                     wgt_we,
    input  wire [WGT_AW-1:0]        wgt_waddr,
    input  wire [COLS*8-1:0]        wgt_wdata,
    input  wire                     bias_we,
    input  wire [BIAS_AW-1:0]       bias_waddr,
    input  wire [COLS*ACC_WIDTH-1:0] bias_wdata,
    input  wire                     act_we,
    input  wire [ACT_AW-1:0]        act_waddr,
    input  wire [ROWS*8-1:0]        act_wdata,

    // activation read port (host, idle only; registered, 1-cycle latency)
    input  wire                     act_re,
    input  wire [ACT_AW-1:0]        act_raddr,
    output wire [ROWS*8-1:0]        act_rdata,

    // performance counters
    output wire [31:0]              perf_cycles,
    output wire [31:0]              perf_act_rd,
    output wire [31:0]              perf_lrb_rd,
    output wire [31:0]              perf_wgt_rd,
    output wire [31:0]              perf_vec
);

    // synopsys translate_off
    initial begin
        if (ROWS != COLS) begin
            $display("ERROR: dnn_accel_core requires ROWS == COLS"); $finish; end
        if (ACC_WIDTH < 16 + $clog2(ROWS*KT_MAX) + 1) begin
            $display("ERROR: ACC_WIDTH=%0d too small for K accumulation (need >= %0d)",
                     ACC_WIDTH, 16 + $clog2(ROWS*KT_MAX) + 1); $finish; end
    end
    // synopsys translate_on

    //-------------------------------------------------------------------
    // Controller
    //-------------------------------------------------------------------
    wire                 wgt_rd_en, wgt_load;
    wire [WGT_AW-1:0]    wgt_rd_addr;
    wire                 act_rd_en;
    wire [ACT_AW-1:0]    act_rd_addr;
    wire                 lrb_rd_en, lrb_wr_en, feed_vld, feed_sel_lrb;
    wire [LRB_AW-1:0]    lrb_rd_addr, lrb_wr_addr;
    wire                 bias_rd_en;
    wire [BIAS_AW-1:0]   bias_rd_addr;
    wire                 pass_start, first_kt, last_kt, ppu_relu, wr_done;
    wire [ACC_AW-1:0]    acc_base;
    wire [ACT_AW-1:0]    dst_start;
    wire [NT_W-1:0]      dst_step;
    wire [15:0]          ppu_mult;
    wire [5:0]           ppu_shift;

    dnn_ctrl #(
        .ROWS(ROWS), .ACC_DEPTH(ACC_DEPTH), .LRB_DEPTH(LRB_DEPTH),
        .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX), .KT_MAX(KT_MAX), .NT_MAX(NT_MAX),
        .M_W(M_W), .KT_W(KT_W), .NT_W(NT_W), .ACT_AW(ACT_AW), .WGT_AW(WGT_AW),
        .BIAS_AW(BIAS_AW), .ACC_AW(ACC_AW), .LRB_AW(LRB_AW), .LAY_W(LAY_W),
        .DESC_AW(DESC_AW)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .start(start), .n_layers(n_layers), .busy(busy), .done(done), .err(err),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .wgt_rd_en(wgt_rd_en), .wgt_rd_addr(wgt_rd_addr), .wgt_load(wgt_load),
        .act_rd_en(act_rd_en), .act_rd_addr(act_rd_addr),
        .lrb_rd_en(lrb_rd_en), .lrb_rd_addr(lrb_rd_addr),
        .lrb_wr_en(lrb_wr_en), .lrb_wr_addr(lrb_wr_addr),
        .feed_vld(feed_vld), .feed_sel_lrb(feed_sel_lrb),
        .bias_rd_en(bias_rd_en), .bias_rd_addr(bias_rd_addr),
        .pass_start(pass_start), .acc_base(acc_base),
        .dst_start(dst_start), .dst_step(dst_step),
        .first_kt(first_kt), .last_kt(last_kt),
        .ppu_mult(ppu_mult), .ppu_shift(ppu_shift), .ppu_relu(ppu_relu),
        .wr_done(wr_done)
    );

    //-------------------------------------------------------------------
    // Memories
    //-------------------------------------------------------------------
    wire [COLS*8-1:0]          wgt_rd_data;
    wire [ROWS*8-1:0]          act_rd_data;
    wire [COLS*ACC_WIDTH-1:0]  bias_rd_data, acc_rd_data;

    sram_sdp #(.WIDTH(COLS*8), .DEPTH(WGT_DEPTH), .AW(WGT_AW)) u_wgt_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en(wgt_we & ~busy), .wr_addr(wgt_waddr), .wr_data(wgt_wdata),
        .rd_en(wgt_rd_en), .rd_addr(wgt_rd_addr), .rd_data(wgt_rd_data));

    sram_sdp #(.WIDTH(COLS*ACC_WIDTH), .DEPTH(BIAS_DEPTH), .AW(BIAS_AW)) u_bias_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en(bias_we & ~busy), .wr_addr(bias_waddr), .wr_data(bias_wdata),
        .rd_en(bias_rd_en), .rd_addr(bias_rd_addr), .rd_data(bias_rd_data));

    // Activation SRAM: PPU owns the write port and the controller the read port
    // while busy; the host owns both while idle.
    wire                 ppu_out_vld;
    wire [COLS*8-1:0]    ppu_out_data;
    reg  [ACT_AW-1:0]    dst_ptr;

    wire                 a_wr_en   = busy ? ppu_out_vld : act_we;
    wire [ACT_AW-1:0]    a_wr_addr = busy ? dst_ptr     : act_waddr;
    wire [ROWS*8-1:0]    a_wr_data = busy ? ppu_out_data: act_wdata;
    wire                 a_rd_en   = busy ? act_rd_en   : act_re;
    wire [ACT_AW-1:0]    a_rd_addr = busy ? act_rd_addr : act_raddr;

    sram_sdp #(.WIDTH(ROWS*8), .DEPTH(ACT_DEPTH), .AW(ACT_AW)) u_act_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en(a_wr_en), .wr_addr(a_wr_addr), .wr_data(a_wr_data),
        .rd_en(a_rd_en), .rd_addr(a_rd_addr), .rd_data(act_rd_data));
    assign act_rdata = act_rd_data;

    //-------------------------------------------------------------------
    // LRB + compute core
    //-------------------------------------------------------------------
    wire                       core_act_vld, core_res_vld, core_res_early;
    wire [ROWS*8-1:0]          core_act_vec;
    wire [COLS*ACC_WIDTH-1:0]  core_res_vec;

    lrb #(.ROWS(ROWS), .DATA_WIDTH(8), .LRB_DEPTH(LRB_DEPTH), .AW(LRB_AW)) u_lrb (
        .clk(clk), .rst_n(rst_n), .en(busy),
        .act_sram_data(act_rd_data),
        .wr_en(lrb_wr_en), .wr_addr(lrb_wr_addr),
        .rd_en(lrb_rd_en), .rd_addr(lrb_rd_addr),
        .feed_vld(feed_vld), .feed_sel_lrb(feed_sel_lrb),
        .act_vld(core_act_vld), .act_vec(core_act_vec));

    sa_core #(.ROWS(ROWS), .COLS(COLS), .DATA_WIDTH(8), .ACC_WIDTH(ACC_WIDTH)) u_core (
        .clk(clk), .rst_n(rst_n), .en(busy),
        .wgt_load(wgt_load), .wgt_in(wgt_rd_data),
        .act_vld(core_act_vld), .act_vec(core_act_vec),
        .res_vld(core_res_vld), .res_vld_early(core_res_early),
        .res_vec(core_res_vec));

    //-------------------------------------------------------------------
    // K-tile accumulation (read-modify-write in acc SRAM)
    //-------------------------------------------------------------------
    reg  [ACC_AW-1:0] acc_rptr, acc_wptr;
    wire              acc_rd_en = core_res_early & ~first_kt;

    // Declared before use (an undeclared name in a port connection would
    // silently become a 1-bit implicit net).
    reg                        s1_vld;
    reg  [COLS*ACC_WIDTH-1:0]  s1_sum;
    wire                       acc_wr_en  = s1_vld & ~last_kt;   // partial -> acc SRAM
    wire                       ppu_in_vld = s1_vld &  last_kt;   // final   -> PPU

    sram_sdp #(.WIDTH(COLS*ACC_WIDTH), .DEPTH(ACC_DEPTH), .AW(ACC_AW)) u_acc_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en(acc_wr_en), .wr_addr(acc_wptr), .wr_data(s1_sum),
        .rd_en(acc_rd_en), .rd_addr(acc_rptr), .rd_data(acc_rd_data));

    // Lane-wise add: this pass's result + running partial sum (0 on first K).
    genvar l;
    generate
        for (l = 0; l < COLS; l = l + 1) begin : g_acc
            wire signed [ACC_WIDTH-1:0] r = core_res_vec[l*ACC_WIDTH +: ACC_WIDTH];
            wire signed [ACC_WIDTH-1:0] o = first_kt ? {ACC_WIDTH{1'b0}}
                                                     : acc_rd_data[l*ACC_WIDTH +: ACC_WIDTH];
            always @(posedge clk) begin
                if (!rst_n)    s1_sum[l*ACC_WIDTH +: ACC_WIDTH] <= {ACC_WIDTH{1'b0}};
                else if (busy) s1_sum[l*ACC_WIDTH +: ACC_WIDTH] <= r + o;
            end
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n)    s1_vld <= 1'b0;
        else if (busy) s1_vld <= core_res_vld;
    end

    //-------------------------------------------------------------------
    // Pointers (reloaded at the start of every pass)
    //-------------------------------------------------------------------
    wire [ACT_AW-1:0] dst_step_ext = dst_step;
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_rptr <= {ACC_AW{1'b0}}; acc_wptr <= {ACC_AW{1'b0}};
            dst_ptr  <= {ACT_AW{1'b0}};
        end else if (pass_start) begin
            acc_rptr <= acc_base; acc_wptr <= acc_base; dst_ptr <= dst_start;
        end else begin
            if (acc_rd_en)   acc_rptr <= acc_rptr + 1'b1;
            if (acc_wr_en)   acc_wptr <= acc_wptr + 1'b1;
            if (ppu_out_vld) dst_ptr  <= dst_ptr + dst_step_ext;
        end
    end

    //-------------------------------------------------------------------
    // Post-processing unit
    //-------------------------------------------------------------------
    dnn_ppu #(.LANES(COLS), .ACC_WIDTH(ACC_WIDTH), .MULT_W(16), .SHIFT_W(6)) u_ppu (
        .clk(clk), .rst_n(rst_n), .en(busy),
        .in_vld(ppu_in_vld), .in_data(s1_sum), .bias(bias_rd_data),
        .mult(ppu_mult), .shift(ppu_shift), .relu(ppu_relu),
        .out_vld(ppu_out_vld), .out_data(ppu_out_data));

    assign wr_done = acc_wr_en | ppu_out_vld;

    //-------------------------------------------------------------------
    // Performance counters
    //-------------------------------------------------------------------
    generate
        if (PERF_EN != 0) begin : g_perf
            perf_counters u_perf (
                .clk(clk), .rst_n(rst_n), .clear(start & ~busy), .busy(busy),
                .act_rd(act_rd_en), .lrb_rd(lrb_rd_en), .wgt_rd(wgt_rd_en),
                .cycles(perf_cycles), .act_sram_rd(perf_act_rd),
                .lrb_rd_cnt(perf_lrb_rd), .wgt_sram_rd(perf_wgt_rd));
            reg [31:0] vec_cnt;
            always @(posedge clk) begin
                if (!rst_n || (start & ~busy)) vec_cnt <= 32'd0;
                else if (feed_vld)             vec_cnt <= vec_cnt + 32'd1;
            end
            assign perf_vec = vec_cnt;
        end else begin : g_noperf
            assign perf_cycles = 32'd0; assign perf_act_rd = 32'd0;
            assign perf_lrb_rd = 32'd0; assign perf_wgt_rd = 32'd0;
            assign perf_vec    = 32'd0;
        end
    endgenerate

endmodule
