//=============================================================================
// Testbench   : tb_systolic_accel
// Description : Self-checking system-level testbench with an in-TB golden
//               model (INT8 x INT8 -> INT32 tiled matmul).
//
//   Covered:
//     - directed corner data: all -128, all +127, -128 x +127, zeros, mixed
//     - shape sweep: M=1, M=ACT_DEPTH, T=1, T=MAX (within OUT_DEPTH)
//     - LRB: full hit (M<=LRB_DEPTH), partial hit (M>LRB_DEPTH), T=1 (no fill)
//       -> exact SRAM/LRB read-count and cycle-count checks
//     - constrained-random regression (NUM_RANDOM runs) back-to-back with no
//       reset in between (proves the pipeline drains clean)
//     - invalid configs rejected (cfg_err, no start)
//     - start pulse while busy ignored; host writes while busy blocked
//     - asynchronous reset in the middle of a run, then recovery
//
//   Run examples (iverilog):
//     iverilog -g2005 -o sim tb/tb_systolic_accel.v rtl/*.v && vvp sim
//     iverilog -g2005 -Ptb_systolic_accel.ROWS=8 -Ptb_systolic_accel.COLS=8 \
//              -Ptb_systolic_accel.OUT_DEPTH=64 ... (see scripts/run_regress.sh)
//=============================================================================
`timescale 1ns/1ps

module tb_systolic_accel;

    parameter ROWS       = 4;
    parameter COLS       = 4;
    parameter DW         = 8;
    parameter ACC        = 32;
    parameter ACT_DEPTH  = 16;
    parameter WGT_DEPTH  = 16;
    parameter OUT_DEPTH  = 32;
    parameter LRB_DEPTH  = 8;
    parameter NUM_RANDOM = 60;
    parameter SEED       = 32'h1BADB002;

    localparam MAX_TILES = WGT_DEPTH / ROWS;
    localparam M_W    = $clog2(ACT_DEPTH + 1);
    localparam T_W    = $clog2(MAX_TILES + 1);
    localparam ACT_AW = (ACT_DEPTH > 1) ? $clog2(ACT_DEPTH) : 1;
    localparam WGT_AW = (WGT_DEPTH > 1) ? $clog2(WGT_DEPTH) : 1;
    localparam OUT_AW = (OUT_DEPTH > 1) ? $clog2(OUT_DEPTH) : 1;
    localparam LAT    = ROWS + COLS - 1;

    //------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------
    reg                       clk = 1'b0;
    reg                       rst_n_i;
    reg                       start;
    reg  [M_W-1:0]            cfg_m;
    reg  [T_W-1:0]            cfg_tiles;
    wire                      busy, done, cfg_err;
    reg                       host_wgt_we;
    reg  [WGT_AW-1:0]         host_wgt_addr;
    reg  [COLS*DW-1:0]        host_wgt_data;
    reg                       host_act_we;
    reg  [ACT_AW-1:0]         host_act_addr;
    reg  [ROWS*DW-1:0]        host_act_data;
    reg                       host_out_re;
    reg  [OUT_AW-1:0]         host_out_addr;
    wire [COLS*ACC-1:0]       host_out_data;
    wire [31:0]               perf_cycles, perf_act_sram_rd, perf_lrb_rd, perf_wgt_sram_rd;

    systolic_accel_top #(
        .ROWS(ROWS), .COLS(COLS), .DATA_WIDTH(DW), .ACC_WIDTH(ACC),
        .ACT_DEPTH(ACT_DEPTH), .WGT_DEPTH(WGT_DEPTH), .OUT_DEPTH(OUT_DEPTH),
        .LRB_DEPTH(LRB_DEPTH), .PERF_EN(1), .SYNC_RST_IN(1)
    ) dut (
        .clk(clk), .rst_n_i(rst_n_i),
        .start(start), .cfg_m(cfg_m), .cfg_tiles(cfg_tiles),
        .busy(busy), .done(done), .cfg_err(cfg_err),
        .host_wgt_we(host_wgt_we), .host_wgt_addr(host_wgt_addr), .host_wgt_data(host_wgt_data),
        .host_act_we(host_act_we), .host_act_addr(host_act_addr), .host_act_data(host_act_data),
        .host_out_re(host_out_re), .host_out_addr(host_out_addr), .host_out_data(host_out_data),
        .perf_cycles(perf_cycles), .perf_act_sram_rd(perf_act_sram_rd),
        .perf_lrb_rd(perf_lrb_rd), .perf_wgt_sram_rd(perf_wgt_sram_rd)
    );

    always #5 clk = ~clk;                       // 100 MHz

    //------------------------------------------------------------------
    // Golden-model storage
    //------------------------------------------------------------------
    reg signed [DW-1:0] A_mat [0:ACT_DEPTH*ROWS-1];          // A[m][r]
    reg signed [DW-1:0] W_mat [0:WGT_DEPTH*COLS-1];          // W[tile*ROWS+r][c]
    integer             EXP   [0:OUT_DEPTH*COLS-1];          // C[tile*M+m][c]

    integer errors    = 0;
    integer seed      = SEED;
    integer test_no   = 0;
    integer total_macs = 0;
    integer total_cyc  = 0;

    //------------------------------------------------------------------
    // Helpers
    //------------------------------------------------------------------
    function integer rnd_range;                 // inclusive [lo, hi]
        input integer lo, hi;
        integer r;
        begin
            r = $random(seed);
            if (r < 0) r = -r;
            if (r < 0) r = 0;                   // -2^31 corner
            rnd_range = lo + (r % (hi - lo + 1));
        end
    endfunction

    function integer imax; input integer a, b; begin imax = (a > b) ? a : b; end endfunction
    function integer imin; input integer a, b; begin imin = (a < b) ? a : b; end endfunction

    task tick; begin @(posedge clk); #1; end endtask

    task do_reset;
        begin
            rst_n_i = 1'b0;
            start = 0; host_wgt_we = 0; host_act_we = 0; host_out_re = 0;
            cfg_m = 0; cfg_tiles = 0;
            repeat (4) tick;
            rst_n_i = 1'b1;
            repeat (4) tick;
        end
    endtask

    // Data patterns: 0 random, 1 all -128, 2 all +127, 3 zeros,
    //                4 A=-128 / W=+127, 5 random extremes {-128,-1,0,1,127}
    task fill_data;
        input integer mode;
        integer i, e;
        begin
            for (i = 0; i < ACT_DEPTH*ROWS; i = i + 1) begin
                case (mode)
                    1:       A_mat[i] = -128;
                    2:       A_mat[i] = 127;
                    3:       A_mat[i] = 0;
                    4:       A_mat[i] = -128;
                    5:       begin e = rnd_range(0,4);
                                   A_mat[i] = (e==0)?-128:(e==1)?-1:(e==2)?0:(e==3)?1:127; end
                    default: A_mat[i] = rnd_range(-128, 127);
                endcase
            end
            for (i = 0; i < WGT_DEPTH*COLS; i = i + 1) begin
                case (mode)
                    1:       W_mat[i] = -128;
                    2:       W_mat[i] = 127;
                    3:       W_mat[i] = 0;
                    4:       W_mat[i] = 127;
                    5:       begin e = rnd_range(0,4);
                                   W_mat[i] = (e==0)?-128:(e==1)?-1:(e==2)?0:(e==3)?1:127; end
                    default: W_mat[i] = rnd_range(-128, 127);
                endcase
            end
        end
    endtask

    task compute_golden;
        input integer M, T;
        integer t, m, c, r, acc;
        begin
            for (t = 0; t < T; t = t + 1)
                for (m = 0; m < M; m = m + 1)
                    for (c = 0; c < COLS; c = c + 1) begin
                        acc = 0;
                        for (r = 0; r < ROWS; r = r + 1)
                            acc = acc + A_mat[m*ROWS + r] * W_mat[(t*ROWS + r)*COLS + c];
                        EXP[(t*M + m)*COLS + c] = acc;
                    end
        end
    endtask

    task load_mems;
        input integer M, T;
        integer m, r, c, w;
        reg [ROWS*DW-1:0] aw;
        reg [COLS*DW-1:0] ww;
        begin
            for (m = 0; m < M; m = m + 1) begin
                for (r = 0; r < ROWS; r = r + 1) aw[r*DW +: DW] = A_mat[m*ROWS + r];
                host_act_we = 1; host_act_addr = m; host_act_data = aw; tick;
            end
            host_act_we = 0;
            for (w = 0; w < T*ROWS; w = w + 1) begin
                for (c = 0; c < COLS; c = c + 1) ww[c*DW +: DW] = W_mat[w*COLS + c];
                host_wgt_we = 1; host_wgt_addr = w; host_wgt_data = ww; tick;
            end
            host_wgt_we = 0;
        end
    endtask

    task pulse_start;
        input integer M, T;
        begin
            cfg_m = M; cfg_tiles = T;
            start = 1; tick; start = 0;
        end
    endtask

    // Wait for a run to finish (done sticky, busy low). 2 idle cycles first so
    // an old sticky `done` cannot be mistaken for the new one.
    task wait_done;
        input integer limit;
        integer t;
        begin
            repeat (2) tick;
            t = 0;
            while ((busy || !done) && t < limit) begin tick; t = t + 1; end
            if (t >= limit) begin
                $display("  [FAIL] timeout waiting for done (test %0d)", test_no);
                errors = errors + 1;
            end
        end
    endtask

    task check_results;
        input integer M, T;
        integer i, c, bad;
        reg [COLS*ACC-1:0] d;
        reg signed [ACC-1:0] got;
        begin
            bad = 0;
            for (i = 0; i < M*T; i = i + 1) begin
                host_out_re = 1; host_out_addr = i; tick;
                host_out_re = 0; d = host_out_data;
                for (c = 0; c < COLS; c = c + 1) begin
                    got = d[c*ACC +: ACC];
                    if (got !== EXP[i*COLS + c]) begin
                        if (bad < 6)
                            $display("  [MISMATCH] test %0d row %0d col %0d: got %0d exp %0d",
                                     test_no, i, c, got, EXP[i*COLS + c]);
                        bad = bad + 1;
                    end
                end
            end
            if (bad != 0) errors = errors + bad;
        end
    endtask

    // Exact expectations for the reuse buffer and the schedule.
    task check_perf;
        input integer M, T;
        integer exp_act, exp_lrb, exp_wgt, exp_cyc;
        begin
            exp_wgt = T * ROWS;
            exp_act = M + (T - 1) * imax(0, M - LRB_DEPTH);
            exp_lrb = (T - 1) * imin(M, LRB_DEPTH);
            exp_cyc = T * (ROWS + M + LAT + 3);         // LDW + COMP + DRAIN per tile
            if (perf_wgt_sram_rd !== exp_wgt) begin
                $display("  [FAIL] wgt reads %0d exp %0d", perf_wgt_sram_rd, exp_wgt); errors = errors + 1; end
            if (perf_act_sram_rd !== exp_act) begin
                $display("  [FAIL] act SRAM reads %0d exp %0d", perf_act_sram_rd, exp_act); errors = errors + 1; end
            if (perf_lrb_rd !== exp_lrb) begin
                $display("  [FAIL] LRB reads %0d exp %0d", perf_lrb_rd, exp_lrb); errors = errors + 1; end
            if (perf_cycles !== exp_cyc) begin
                $display("  [FAIL] cycles %0d exp %0d", perf_cycles, exp_cyc); errors = errors + 1; end
        end
    endtask

    // One complete test: fill, golden, load, run, check.
    //   disturb=1: extra start pulse + host writes while busy (must be ignored)
    task run_case;
        input integer mode, M, T, disturb;
        integer errs_before;
        begin
            test_no     = test_no + 1;
            errs_before = errors;
            fill_data(mode);
            compute_golden(M, T);
            load_mems(M, T);
            pulse_start(M, T);
            if (disturb) begin
                repeat (3) tick;
                // second start while busy: must be ignored
                start = 1; cfg_m = 1; cfg_tiles = 1; tick; start = 0;
                // host writes while busy: must be blocked
                host_act_we = 1; host_act_addr = 0; host_act_data = {ROWS*DW{1'b1}};
                host_wgt_we = 1; host_wgt_addr = 0; host_wgt_data = {COLS*DW{1'b1}};
                repeat (4) tick;
                host_act_we = 0; host_wgt_we = 0;
            end
            wait_done(20000);
            if (cfg_err) begin $display("  [FAIL] unexpected cfg_err"); errors = errors + 1; end
            check_results(M, T);
            check_perf(M, T);
            total_macs = total_macs + M*T*ROWS*COLS;
            total_cyc  = total_cyc + perf_cycles;
            $display("test %3d mode=%0d M=%2d T=%0d dist=%0d | cyc=%4d actRd=%3d lrbRd=%3d wgtRd=%3d | %s",
                     test_no, mode, M, T, disturb, perf_cycles, perf_act_sram_rd,
                     perf_lrb_rd, perf_wgt_sram_rd, (errors == errs_before) ? "PASS" : "FAIL");
        end
    endtask

    // Invalid config must be rejected with cfg_err and no activity.
    task run_bad_cfg;
        input integer M, T;
        begin
            test_no = test_no + 1;
            cfg_m = M; cfg_tiles = T;
            start = 1; tick; start = 0;
            repeat (3) tick;
            if (!cfg_err || busy || done) begin
                $display("test %3d bad cfg M=%0d T=%0d | [FAIL] cfg_err=%b busy=%b done=%b",
                         test_no, M, T, cfg_err, busy, done);
                errors = errors + 1;
            end else
                $display("test %3d bad cfg M=%0d T=%0d rejected | PASS", test_no, M, T);
        end
    endtask

    //------------------------------------------------------------------
    // Main sequence
    //------------------------------------------------------------------
    integer i, M, T, mode, maxT;
    real util;

    initial begin
        $display("=== tb_systolic_accel: %0dx%0d array, ACT=%0d WGT=%0d OUT=%0d LRB=%0d ===",
                 ROWS, COLS, ACT_DEPTH, WGT_DEPTH, OUT_DEPTH, LRB_DEPTH);
        do_reset;

        // ---- directed data corners (M=4,T=2 exercises LRB replay) ----
        for (mode = 1; mode <= 5; mode = mode + 1)
            run_case(mode, imin(4, ACT_DEPTH), imin(2, MAX_TILES), 0);

        // ---- shape corners ----
        run_case(0, 1, 1, 0);                                        // smallest
        run_case(0, 1, imin(MAX_TILES, OUT_DEPTH), 0);               // M=1, max tiles
        run_case(0, imin(ACT_DEPTH, OUT_DEPTH), 1, 0);               // max M, one tile
        run_case(5, imin(LRB_DEPTH, ACT_DEPTH), imin(MAX_TILES, OUT_DEPTH/imin(LRB_DEPTH,ACT_DEPTH)), 0); // full LRB hit
        if (ACT_DEPTH > LRB_DEPTH)                                   // partial LRB hit
            run_case(0, imin(ACT_DEPTH, LRB_DEPTH+4), imin(2, MAX_TILES), 0);
        run_case(0, imin(ACT_DEPTH, OUT_DEPTH/imin(MAX_TILES,OUT_DEPTH)), imin(MAX_TILES, OUT_DEPTH), 0); // max T

        // ---- disturbances: start-while-busy, writes-while-busy ----
        run_case(0, imin(8, ACT_DEPTH), imin(2, MAX_TILES), 1);

        // ---- invalid configs ----
        run_bad_cfg(0, 1);                                           // M = 0
        run_bad_cfg(1, 0);                                           // T = 0
        if ((1 << M_W) - 1 > ACT_DEPTH)  run_bad_cfg((1 << M_W) - 1, 1);   // M too big
        if ((1 << T_W) - 1 > MAX_TILES)  run_bad_cfg(1, (1 << T_W) - 1);   // T too big
        if (ACT_DEPTH * MAX_TILES > OUT_DEPTH)
            run_bad_cfg(ACT_DEPTH, MAX_TILES);                       // exceeds OUT_DEPTH
        // a good run right after the rejects clears cfg_err
        run_case(0, 3, 1, 0);
        if (cfg_err) begin $display("  [FAIL] cfg_err not cleared by good start"); errors = errors + 1; end

        // ---- constrained-random regression, back to back, no reset ----
        for (i = 0; i < NUM_RANDOM; i = i + 1) begin
            M = rnd_range(1, ACT_DEPTH);
            maxT = imin(MAX_TILES, OUT_DEPTH / M);
            T = rnd_range(1, maxT);
            mode = (rnd_range(0, 9) == 0) ? 5 : 0;
            run_case(mode, M, T, (rnd_range(0, 7) == 0));
        end

        // ---- asynchronous reset in the middle of a run, then recover ----
        test_no = test_no + 1;
        fill_data(0); compute_golden(8, 2); load_mems(8, 2);
        pulse_start(8, 2);
        repeat (12) tick;
        #3 rst_n_i = 1'b0;                                          // mid-cycle, async
        repeat (3) tick;
        rst_n_i = 1'b1;
        repeat (4) tick;
        if (busy || done || cfg_err) begin
            $display("test %3d reset-in-run | [FAIL] busy=%b done=%b cfg_err=%b", test_no, busy, done, cfg_err);
            errors = errors + 1;
        end else
            $display("test %3d reset-in-run returns to idle | PASS", test_no);
        run_case(0, 8, 2, 0);                                        // must work again

        //--------------------------------------------------------------
        util = (total_cyc > 0) ? (100.0 * total_macs) / (1.0 * total_cyc * ROWS * COLS) : 0.0;
        $display("---------------------------------------------------------------");
        $display("tests run: %0d   total MACs: %0d   busy cycles: %0d   avg array utilization: %0.1f %%",
                 test_no, total_macs, total_cyc, util);
        if (errors == 0) $display("*** TB PASSED ***");
        else             $display("*** TB FAILED: %0d error(s) ***", errors);
        $finish;
    end

    // Global watchdog
    initial begin
        #200_000_000;
        $display("*** TB FAILED: global watchdog ***");
        $finish;
    end

endmodule
