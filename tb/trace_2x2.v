//=============================================================================
// trace_2x2 : prints a cycle-by-cycle trace of the REAL RTL for a tiny case,
// used in docs/DESIGN_NOTES.md to explain skew / dataflow.
//     A (M=2,K=2) = [[1,2],[3,4]]    W (K=2,N=2) = [[5,6],[7,8]]
//     C = A x W   = [[19,22],[43,50]]
//   iverilog -g2005 -o tr tb/trace_2x2.v rtl/*.v && vvp tr
//=============================================================================
`timescale 1ns/1ps
module trace_2x2;
    reg clk = 0, rst_n_i = 0, start = 0;
    reg  [2:0]  cfg_m = 0;      // M_W = clog2(4+1) = 3
    reg  [1:0]  cfg_tiles = 0;  // T_W = clog2(4/2+1) = 2
    wire busy, done, cfg_err;
    reg  host_wgt_we = 0, host_act_we = 0, host_out_re = 0;
    reg  [1:0]  host_wgt_addr = 0;
    reg  [15:0] host_wgt_data = 0;
    reg  [1:0]  host_act_addr = 0;
    reg  [15:0] host_act_data = 0;
    reg  [2:0]  host_out_addr = 0;
    wire [63:0] host_out_data;
    wire [31:0] p0, p1, p2, p3;

    systolic_accel_top #(.ROWS(2), .COLS(2), .ACT_DEPTH(4), .WGT_DEPTH(4),
                         .OUT_DEPTH(8), .LRB_DEPTH(2)) dut (
        .clk(clk), .rst_n_i(rst_n_i), .start(start), .cfg_m(cfg_m), .cfg_tiles(cfg_tiles),
        .busy(busy), .done(done), .cfg_err(cfg_err),
        .host_wgt_we(host_wgt_we), .host_wgt_addr(host_wgt_addr), .host_wgt_data(host_wgt_data),
        .host_act_we(host_act_we), .host_act_addr(host_act_addr), .host_act_data(host_act_data),
        .host_out_re(host_out_re), .host_out_addr(host_out_addr), .host_out_data(host_out_data),
        .perf_cycles(p0), .perf_act_sram_rd(p1), .perf_lrb_rd(p2), .perf_wgt_sram_rd(p3));

    always #5 clk = ~clk;

    // Peek at internal nets
    wire signed [7:0]  a_r0 = dut.u_core.act_skewed[7:0];      // act into PE(0,0)
    wire signed [7:0]  a_r1 = dut.u_core.act_skewed[15:8];     // act into PE(1,0)
    wire signed [7:0]  w00 = dut.u_core.u_array.g_row[0].g_col[0].u_pe.weight_reg;
    wire signed [7:0]  w01 = dut.u_core.u_array.g_row[0].g_col[1].u_pe.weight_reg;
    wire signed [7:0]  w10 = dut.u_core.u_array.g_row[1].g_col[0].u_pe.weight_reg;
    wire signed [7:0]  w11 = dut.u_core.u_array.g_row[1].g_col[1].u_pe.weight_reg;
    wire signed [31:0] pb0 = dut.u_core.psum_raw[31:0];        // bottom psum col 0
    wire signed [31:0] pb1 = dut.u_core.psum_raw[63:32];       // bottom psum col 1
    wire signed [31:0] r0  = dut.u_core.res_vec[31:0];         // deskewed
    wire signed [31:0] r1  = dut.u_core.res_vec[63:32];

    integer cyc = 0;
    reg started = 0;
    always @(posedge clk) begin
        if (started) begin
            #1;
            $display("%3d | st=%0d wl=%b | actvld=%b vec=[%0d,%0d] | skewed rows=[%0d,%0d] | W=[[%0d,%0d],[%0d,%0d]] | botpsum=[%0d,%0d] | res_vld=%b res=[%0d,%0d]",
                cyc, dut.u_ctrl.state, dut.wgt_load, dut.core_act_vld,
                $signed(dut.core_act_vec[7:0]), $signed(dut.core_act_vec[15:8]),
                a_r0, a_r1, w00, w01, w10, w11, pb0, pb1, dut.core_res_vld, r0, r1);
            cyc = cyc + 1;
        end
    end

    task tick; begin @(posedge clk); #1; end endtask

    initial begin
        repeat (3) tick; rst_n_i = 1; repeat (3) tick;
        // Activations: word = {A[m][1], A[m][0]}
        host_act_we = 1; host_act_addr = 0; host_act_data = {8'sd2, 8'sd1}; tick;   // A[0] = [1,2]
                         host_act_addr = 1; host_act_data = {8'sd4, 8'sd3}; tick;   // A[1] = [3,4]
        host_act_we = 0;
        // Weights: word = {W[r][1], W[r][0]}, address = r
        host_wgt_we = 1; host_wgt_addr = 0; host_wgt_data = {8'sd6, 8'sd5}; tick;   // W[0] = [5,6]
                         host_wgt_addr = 1; host_wgt_data = {8'sd8, 8'sd7}; tick;   // W[1] = [7,8]
        host_wgt_we = 0;
        cfg_m = 2; cfg_tiles = 1;
        $display("cyc | state wgt_load | act into core | skewed act rows | PE weights | bottom psums | deskewed result");
        start = 1; started = 1; tick; start = 0;
        while (!done || busy) tick;
        repeat (2) tick;
        host_out_re = 1; host_out_addr = 0; tick; $display("out[0] = [%0d, %0d]  (expect 19, 22)",
            $signed(host_out_data[31:0]), $signed(host_out_data[63:32]));
        host_out_addr = 1; tick; $display("out[1] = [%0d, %0d]  (expect 43, 50)",
            $signed(host_out_data[31:0]), $signed(host_out_data[63:32]));
        $finish;
    end
endmodule
