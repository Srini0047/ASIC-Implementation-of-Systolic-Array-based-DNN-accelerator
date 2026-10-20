//=============================================================================
// Module      : systolic_accel_top
// Description : Weight-stationary INT8 systolic-array DNN accelerator with a
//               Local Reuse Buffer (LRB). Computes tiled matrix products
//                   C[M x (T*COLS)] = A[M x ROWS] * W[ROWS x (T*COLS)]
//               (INT8 x INT8 -> INT32), the core of DNN inference layers.
//
//   Hierarchy
//     systolic_accel_top
//       |- rst_sync          async-assert / sync-release reset
//       |- accel_ctrl        FSM: IDLE -> LDW -> COMP -> DRAIN (per tile)
//       |- sram_sdp x3       u_wgt_mem, u_act_mem, u_out_mem (1R1W, 1-cycle read)
//       |- lrb               local reuse buffer + source mux + feed register
//       |     '- sram_sdp    LRB storage
//       |- sa_core           compute core
//       |     |- skew_delay  input skew  (row r delayed r)
//       |     |- systolic_array
//       |     |     '- pe x (ROWS*COLS)
//       |     '- skew_delay  output deskew (col c delayed COLS-1-c)
//       '- perf_counters     optional (PERF_EN)
//
//   Host usage
//     1. Write weights: word = COLS INT8 weights, address = tile*ROWS + row.
//        weight[r][c] occupies host_wgt_data[c*DW +: DW].
//     2. Write activations: word = ROWS INT8 values, address = m.
//        A[m][r] occupies host_act_data[r*DW +: DW].
//     3. Set cfg_m (1..ACT_DEPTH) and cfg_tiles (1..WGT_DEPTH/ROWS) with
//        tiles*M <= OUT_DEPTH, pulse `start` while !busy.
//     4. Wait for `done` (sticky). Read results: address = tile*M + m,
//        word = COLS INT32 values, C[m][tile*COLS + c] = data[c*ACC +: ACC].
//        Memory writes are ignored while busy.
//
//   All host inputs must be synchronous to clk. rst_n_i may be asynchronous
//   (see SYNC_RST_IN).
//
//   Parameters that end in _W / _AW are DERIVED - do not override them.
//=============================================================================
module systolic_accel_top #(
    parameter ROWS        = 4,       // K per pass = array rows
    parameter COLS        = 4,       // outputs per pass = array columns
    parameter DATA_WIDTH  = 8,       // INT8
    parameter ACC_WIDTH   = 32,      // INT32 accumulate
    parameter ACT_DEPTH   = 16,      // activation vectors stored (max M)
    parameter WGT_DEPTH   = 16,      // weight rows stored (= ROWS * max tiles)
    parameter OUT_DEPTH   = 32,      // result vectors stored (>= M*tiles)
    parameter LRB_DEPTH   = 8,       // activation vectors cached in the LRB
    parameter PERF_EN     = 1,       // 1: include performance counters
    parameter SYNC_RST_IN = 1,       // 1: run rst_n_i through rst_sync

    // ---- derived widths (do not override) ----
    parameter M_W    = $clog2(ACT_DEPTH + 1),
    parameter T_W    = $clog2(WGT_DEPTH / ROWS + 1),
    parameter ACT_AW = (ACT_DEPTH > 1) ? $clog2(ACT_DEPTH) : 1,
    parameter WGT_AW = (WGT_DEPTH > 1) ? $clog2(WGT_DEPTH) : 1,
    parameter OUT_AW = (OUT_DEPTH > 1) ? $clog2(OUT_DEPTH) : 1,
    parameter LRB_AW = (LRB_DEPTH > 1) ? $clog2(LRB_DEPTH) : 1
)(
    input  wire                         clk,
    input  wire                         rst_n_i,      // active low

    // ---- control / status ----
    input  wire                         start,
    input  wire [M_W-1:0]               cfg_m,
    input  wire [T_W-1:0]               cfg_tiles,
    output wire                         busy,
    output wire                         done,
    output wire                         cfg_err,

    // ---- host write: weights ----
    input  wire                         host_wgt_we,
    input  wire [WGT_AW-1:0]            host_wgt_addr,
    input  wire [COLS*DATA_WIDTH-1:0]   host_wgt_data,

    // ---- host write: activations ----
    input  wire                         host_act_we,
    input  wire [ACT_AW-1:0]            host_act_addr,
    input  wire [ROWS*DATA_WIDTH-1:0]   host_act_data,

    // ---- host read: results (registered, 1-cycle latency) ----
    input  wire                         host_out_re,
    input  wire [OUT_AW-1:0]            host_out_addr,
    output wire [COLS*ACC_WIDTH-1:0]    host_out_data,

    // ---- performance counters (0 when PERF_EN=0) ----
    output wire [31:0]                  perf_cycles,
    output wire [31:0]                  perf_act_sram_rd,
    output wire [31:0]                  perf_lrb_rd,
    output wire [31:0]                  perf_wgt_sram_rd
);

    //-------------------------------------------------------------------
    // Elaboration-time sanity checks (simulation only)
    //-------------------------------------------------------------------
    // synopsys translate_off
    initial begin
        if (ACC_WIDTH < 2*DATA_WIDTH + $clog2(ROWS)) begin
            $display("ERROR: ACC_WIDTH=%0d may overflow (need >= %0d)",
                     ACC_WIDTH, 2*DATA_WIDTH + $clog2(ROWS));
            $finish;
        end
        if (WGT_DEPTH < ROWS) begin
            $display("ERROR: WGT_DEPTH must be >= ROWS");
            $finish;
        end
        if (WGT_DEPTH % ROWS != 0)
            $display("WARNING: WGT_DEPTH is not a multiple of ROWS; last %0d rows unusable",
                     WGT_DEPTH % ROWS);
    end
    // synopsys translate_on

    //-------------------------------------------------------------------
    // Reset
    //-------------------------------------------------------------------
    wire rst_n;
    generate
        if (SYNC_RST_IN != 0) begin : g_rst_sync
            rst_sync #(.STAGES(2)) u_rst_sync (
                .clk    (clk),
                .arst_n (rst_n_i),
                .srst_n (rst_n)
            );
        end else begin : g_rst_direct
            assign rst_n = rst_n_i;
        end
    endgenerate

    //-------------------------------------------------------------------
    // Controller <-> datapath nets
    //-------------------------------------------------------------------
    wire                     wgt_rd_en, wgt_load;
    wire [WGT_AW-1:0]        wgt_rd_addr;
    wire                     act_rd_en;
    wire [ACT_AW-1:0]        act_rd_addr;
    wire                     lrb_rd_en, lrb_wr_en;
    wire [LRB_AW-1:0]        lrb_rd_addr, lrb_wr_addr;
    wire                     feed_vld, feed_sel_lrb;
    wire                     out_wr_en;
    wire [OUT_AW-1:0]        out_wr_addr;

    wire [COLS*DATA_WIDTH-1:0] wgt_rd_data;
    wire [ROWS*DATA_WIDTH-1:0] act_rd_data;
    wire                       core_act_vld;
    wire [ROWS*DATA_WIDTH-1:0] core_act_vec;
    wire                       core_res_vld;
    /* verilator lint_off UNUSEDSIGNAL */
    wire                       core_res_early_unused;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [COLS*ACC_WIDTH-1:0]  core_res_vec;

    //-------------------------------------------------------------------
    // Controller
    //-------------------------------------------------------------------
    accel_ctrl #(
        .ROWS(ROWS), .ACT_DEPTH(ACT_DEPTH), .WGT_DEPTH(WGT_DEPTH),
        .OUT_DEPTH(OUT_DEPTH), .LRB_DEPTH(LRB_DEPTH),
        .M_W(M_W), .T_W(T_W), .ACT_AW(ACT_AW), .WGT_AW(WGT_AW),
        .OUT_AW(OUT_AW), .LRB_AW(LRB_AW)
    ) u_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .cfg_m        (cfg_m),
        .cfg_tiles    (cfg_tiles),
        .busy         (busy),
        .done         (done),
        .cfg_err      (cfg_err),
        .wgt_rd_en    (wgt_rd_en),
        .wgt_rd_addr  (wgt_rd_addr),
        .wgt_load     (wgt_load),
        .act_rd_en    (act_rd_en),
        .act_rd_addr  (act_rd_addr),
        .lrb_rd_en    (lrb_rd_en),
        .lrb_rd_addr  (lrb_rd_addr),
        .lrb_wr_en    (lrb_wr_en),
        .lrb_wr_addr  (lrb_wr_addr),
        .feed_vld     (feed_vld),
        .feed_sel_lrb (feed_sel_lrb),
        .res_vld      (core_res_vld),
        .out_wr_en    (out_wr_en),
        .out_wr_addr  (out_wr_addr)
    );

    //-------------------------------------------------------------------
    // On-chip memories (host writes are blocked while busy)
    //-------------------------------------------------------------------
    sram_sdp #(.WIDTH(COLS*DATA_WIDTH), .DEPTH(WGT_DEPTH), .AW(WGT_AW)) u_wgt_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en  (host_wgt_we & ~busy),
        .wr_addr(host_wgt_addr),
        .wr_data(host_wgt_data),
        .rd_en  (wgt_rd_en),
        .rd_addr(wgt_rd_addr),
        .rd_data(wgt_rd_data)
    );

    sram_sdp #(.WIDTH(ROWS*DATA_WIDTH), .DEPTH(ACT_DEPTH), .AW(ACT_AW)) u_act_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en  (host_act_we & ~busy),
        .wr_addr(host_act_addr),
        .wr_data(host_act_data),
        .rd_en  (act_rd_en),
        .rd_addr(act_rd_addr),
        .rd_data(act_rd_data)
    );

    sram_sdp #(.WIDTH(COLS*ACC_WIDTH), .DEPTH(OUT_DEPTH), .AW(OUT_AW)) u_out_mem (
        .clk(clk), .rst_n(rst_n),
        .wr_en  (out_wr_en),
        .wr_addr(out_wr_addr),
        .wr_data(core_res_vec),
        .rd_en  (host_out_re),
        .rd_addr(host_out_addr),
        .rd_data(host_out_data)
    );

    //-------------------------------------------------------------------
    // Local Reuse Buffer + activation feed register
    //-------------------------------------------------------------------
    lrb #(.ROWS(ROWS), .DATA_WIDTH(DATA_WIDTH), .LRB_DEPTH(LRB_DEPTH), .AW(LRB_AW)) u_lrb (
        .clk           (clk),
        .rst_n         (rst_n),
        .en            (busy),
        .act_sram_data (act_rd_data),
        .wr_en         (lrb_wr_en),
        .wr_addr       (lrb_wr_addr),
        .rd_en         (lrb_rd_en),
        .rd_addr       (lrb_rd_addr),
        .feed_vld      (feed_vld),
        .feed_sel_lrb  (feed_sel_lrb),
        .act_vld       (core_act_vld),
        .act_vec       (core_act_vec)
    );

    //-------------------------------------------------------------------
    // Compute core (pipeline enabled only while busy)
    //-------------------------------------------------------------------
    sa_core #(
        .ROWS(ROWS), .COLS(COLS),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_core (
        .clk     (clk),
        .rst_n   (rst_n),
        .en      (busy),
        .wgt_load(wgt_load),
        .wgt_in  (wgt_rd_data),
        .act_vld (core_act_vld),
        .act_vec (core_act_vec),
        .res_vld (core_res_vld),
        .res_vld_early (core_res_early_unused),  // only the DNN top needs it
        .res_vec (core_res_vec)
    );

    //-------------------------------------------------------------------
    // Optional performance counters
    //-------------------------------------------------------------------
    generate
        if (PERF_EN != 0) begin : g_perf
            perf_counters u_perf (
                .clk         (clk),
                .rst_n       (rst_n),
                .clear       (start & ~busy),
                .busy        (busy),
                .act_rd      (act_rd_en),
                .lrb_rd      (lrb_rd_en),
                .wgt_rd      (wgt_rd_en),
                .cycles      (perf_cycles),
                .act_sram_rd (perf_act_sram_rd),
                .lrb_rd_cnt  (perf_lrb_rd),
                .wgt_sram_rd (perf_wgt_sram_rd)
            );
        end else begin : g_noperf
            assign perf_cycles      = 32'd0;
            assign perf_act_sram_rd = 32'd0;
            assign perf_lrb_rd      = 32'd0;
            assign perf_wgt_sram_rd = 32'd0;
        end
    endgenerate

endmodule
