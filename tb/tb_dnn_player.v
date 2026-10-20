//=============================================================================
// Testbench   : tb_dnn_player
// Description : Generic "CPU in a box". It replays a register program produced
//               by sw/compile.py (or sw/gen_random_net.py) against dnn_accel_top,
//               exactly as a host driver would, and checks every read.
//
//   Program line (16 hex digits):  OP(8) 00(8) ADDR(16) DATA(32)
//     0x01 WRITE          reg[ADDR] <= DATA
//     0x02 WAIT_DONE      poll STATUS until !busy; require done=1, err=0
//     0x03 READ_EXPECT    read reg[ADDR]; must equal DATA (and is logged)
//     0x04 READ_LOG       read reg[ADDR]; logged only
//     0x05 WAIT_RD_VALID  poll STATUS.rd_valid (act read data ready)
//     0x06 DELAY          wait DATA clock cycles
//     0x07 WAIT_DONE_ERR  poll until !busy; require err=1, done=0
//     0xFF END
//
//   Plusargs: +prog=<file> (default build/prog.hex)  +log=<file> (default
//   build/rtl_readback.txt). Every READ op appends "R <index> <addr> <data>".
//   Parameters must match the RTL build; pass them with -G (Verilator) or -P.
//   Prints "PLAYER PASSED" / "PLAYER FAILED".
//=============================================================================
`timescale 1ns/1ps

module tb_dnn_player;

    parameter ROWS       = 4;
    parameter COLS       = 4;
    parameter ACT_DEPTH  = 256;
    parameter WGT_DEPTH  = 512;
    parameter BIAS_DEPTH = 16;
    parameter ACC_DEPTH  = 32;
    parameter LRB_DEPTH  = 8;
    parameter NUM_LAYERS = 4;
    parameter M_MAX      = 8;
    parameter KT_MAX     = 16;
    parameter NT_MAX     = 8;
    parameter AXIL       = 0;             // 1: drive the DUT through the AXI4-Lite bridge
    parameter PROG_MAX   = 262144;
    parameter WAIT_LIMIT = 4000000;       // cycles allowed per WAIT_DONE

    reg         clk = 1'b0;
    reg         rst_n_i;
    reg         reg_wr_en = 1'b0, reg_rd_en = 1'b0;
    reg  [15:0] reg_addr  = 16'd0;
    reg  [31:0] reg_wdata = 32'd0;
    wire [31:0] reg_rdata;
    wire        irq;

    // ---- AXI4-Lite master signals (used only when AXIL=1) ----
    reg  [17:0] awaddr = 18'd0, araddr = 18'd0;
    reg         awvalid = 1'b0, wvalid = 1'b0, arvalid = 1'b0, bready = 1'b0, rready = 1'b0;
    reg  [31:0] wdata = 32'd0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;

    generate
        if (AXIL != 0) begin : g_axil
            dnn_accel_axil_top #(
                .ROWS(ROWS), .COLS(COLS), .ACT_DEPTH(ACT_DEPTH), .WGT_DEPTH(WGT_DEPTH),
                .BIAS_DEPTH(BIAS_DEPTH), .ACC_DEPTH(ACC_DEPTH), .LRB_DEPTH(LRB_DEPTH),
                .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX), .KT_MAX(KT_MAX), .NT_MAX(NT_MAX)
            ) dut (
                .aclk(clk), .aresetn(rst_n_i),
                .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
                .s_axil_wdata(wdata), .s_axil_wstrb(4'hF), .s_axil_wvalid(wvalid),
                .s_axil_wready(wready),
                .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
                .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
                .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid),
                .s_axil_rready(rready), .irq(irq));
            assign reg_rdata = 32'd0;
        end else begin : g_reg
            dnn_accel_top #(
                .ROWS(ROWS), .COLS(COLS), .ACT_DEPTH(ACT_DEPTH), .WGT_DEPTH(WGT_DEPTH),
                .BIAS_DEPTH(BIAS_DEPTH), .ACC_DEPTH(ACC_DEPTH), .LRB_DEPTH(LRB_DEPTH),
                .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX), .KT_MAX(KT_MAX), .NT_MAX(NT_MAX)
            ) dut (
                .clk(clk), .rst_n_i(rst_n_i), .reg_wr_en(reg_wr_en), .reg_rd_en(reg_rd_en),
                .reg_addr(reg_addr), .reg_wdata(reg_wdata), .reg_rdata(reg_rdata), .irq(irq));
            assign awready = 1'b0; assign wready = 1'b0; assign bvalid = 1'b0;
            assign bresp = 2'b00;  assign arready = 1'b0; assign rvalid = 1'b0;
            assign rresp = 2'b00;  assign rdata = 32'd0;
        end
    endgenerate

    always #5 clk = ~clk;                        // 100 MHz nominal

`ifdef STROBES
    // Optional (build with -DSTROBES, register-bus build only): count the accumulator-SRAM
    // strobes over the whole run. sw/sweep_arch.py and scripts/check_psum_formula.sh compare
    // them with the partial-sum formula  (KT-1) * M * NT  per layer per batch.
    integer stb_wr = 0, stb_rd = 0;
    always @(posedge clk) begin
        if (g_reg.dut.u_core.acc_wr_en) stb_wr = stb_wr + 1;
        if (g_reg.dut.u_core.acc_rd_en) stb_rd = stb_rd + 1;
    end
`endif

    task tick; begin @(posedge clk); #1; end endtask

    // One bus write / read. With AXIL=0 these are the simple register-bus cycles;
    // with AXIL=1 they are complete AXI4-Lite transactions (address+data handshake,
    // write response; address handshake, read data) - what a CPU's bus master does.
    integer stall_en;                            // +axi_stall : random AXI handshake delays

    function integer rnd;                        // 0..3 when stalling, else 0
        input dummy;
        begin rnd = stall_en ? ($urandom % 4) : 0; end
    endfunction

    integer axi_err = 0;

    task bus_write;
        input [15:0] a; input [31:0] d;
        integer cnt, aw_dly, w_dly;
        reg aw_done, w_done, ahs, whs;
        begin
            if (AXIL == 0) begin
                reg_addr = a; reg_wdata = d; reg_wr_en = 1'b1; tick; reg_wr_en = 1'b0;
            end else begin
                // AW and W are INDEPENDENT channels: each master valid drops only after its
                // own handshake, and either one may be presented first.
                repeat (rnd(0)) tick;
                awaddr = {a, 2'b00}; wdata = d;
                aw_dly = rnd(0); w_dly = rnd(0); cnt = 0; aw_done = 1'b0; w_done = 1'b0;
                while (!(aw_done && w_done) && cnt < 1000) begin
                    if (!aw_done && cnt >= aw_dly) awvalid = 1'b1;
                    if (!w_done  && cnt >= w_dly)  wvalid  = 1'b1;
                    #1; ahs = awvalid & awready; whs = wvalid & wready; tick;
                    if (ahs) begin awvalid = 1'b0; aw_done = 1'b1; end
                    if (whs) begin wvalid  = 1'b0; w_done  = 1'b1; end
                    cnt = cnt + 1;
                end
                awvalid = 1'b0; wvalid = 1'b0;
                repeat (rnd(0)) tick;                                    // master slow to accept B
                bready = 1'b1;
                cnt = 0;
                while (!bvalid && cnt < 1000) begin tick; cnt = cnt + 1; end   // write response
                if (!bvalid) begin axi_err = axi_err + 1; $display("[FAIL] AXI write: no B response"); end
                tick; bready = 1'b0;
                if (bresp !== 2'b00) begin axi_err = axi_err + 1; $display("[FAIL] AXI write response %b", bresp); end
            end
        end
    endtask

    task bus_read;                               // data valid when the task returns
        input  [15:0] a; output [31:0] d;
        integer cnt;
        reg hs;
        begin
            if (AXIL == 0) begin
                reg_addr = a; reg_rd_en = 1'b1; tick; reg_rd_en = 1'b0; d = reg_rdata;
            end else begin
                repeat (rnd(0)) tick;
                araddr = {a, 2'b00}; arvalid = 1'b1;
                hs = 1'b0; cnt = 0;
                while (!hs && cnt < 1000) begin #1; hs = arready; tick; cnt = cnt + 1; end
                arvalid = 1'b0;
                repeat (rnd(0)) tick;                                    // master slow to accept R
                rready = 1'b1;
                cnt = 0;
                while (!rvalid && cnt < 1000) begin tick; cnt = cnt + 1; end
                if (!rvalid) begin axi_err = axi_err + 1; $display("[FAIL] AXI read: no R response"); end
                d = rdata;                                               // stable while rvalid && !rready
                tick; rready = 1'b0;
            end
        end
    endtask

    reg [63:0]  prog [0:PROG_MAX-1];
    reg [8*256-1:0] progfile, logfile;
    integer     logfd, pc, nread, mism, timeouts, nops, running;
    integer     cyc_total;
    reg [63:0]  ins;
    reg [7:0]   op;
    reg [15:0]  addr;
    reg [31:0]  data, got, st;
    integer     t;

    initial begin
`ifdef WAVES
        $dumpfile("build/dnn.vcd");            // build with -DWAVES (see scripts/run_wave.sh)
        $dumpvars(0, tb_dnn_player);
`endif
        stall_en = $test$plusargs("axi_stall") ? 1 : 0;
        if (!$value$plusargs("prog=%s", progfile)) progfile = "build/prog.hex";
        if (!$value$plusargs("log=%s",  logfile))  logfile  = "build/rtl_readback.txt";
        for (pc = 0; pc < PROG_MAX; pc = pc + 1) prog[pc] = {8'hFF, 56'd0};
        $readmemh(progfile, prog);
        if (prog[0][63:56] == 8'hFF) begin      // nothing was loaded: a missing/empty file must not look like a pass
            $display("PLAYER: program file empty or not found: %0s", progfile);
            $display("*** PLAYER FAILED ***");
            $finish;
        end
        logfd = $fopen(logfile, "w");

        rst_n_i = 1'b0;
        repeat (4) tick;
        rst_n_i = 1'b1;
        repeat (4) tick;

        pc = 0; nread = 0; mism = 0; timeouts = 0; nops = 0; running = 1; cyc_total = 0;
        while (running) begin
            ins = prog[pc]; pc = pc + 1; nops = nops + 1;
            op = ins[63:56]; addr = ins[47:32]; data = ins[31:0];
            case (op)
                8'h01: bus_write(addr, data);

                8'h02, 8'h07: begin                          // WAIT_DONE / WAIT_DONE_ERR
                    t = 0; st = 32'd1;
                    while ((st[0] || !(st[1] || st[2])) && t < WAIT_LIMIT) begin
                        bus_read(16'h0001, st); t = t + 1;
                    end
                    cyc_total = cyc_total + t;
                    if (t >= WAIT_LIMIT) begin
                        timeouts = timeouts + 1;
                        $display("[FAIL] pc=%0d timeout waiting for done", pc - 1);
                    end else if (op == 8'h02 && !(st[1] && !st[2])) begin
                        mism = mism + 1;
                        $display("[FAIL] pc=%0d expected done, STATUS=%h", pc - 1, st);
                    end else if (op == 8'h07 && !(st[2] && !st[1])) begin
                        mism = mism + 1;
                        $display("[FAIL] pc=%0d expected err, STATUS=%h", pc - 1, st);
                    end
                end

                8'h03, 8'h04: begin                          // READ_EXPECT / READ_LOG
                    bus_read(addr, got);
                    $fdisplay(logfd, "R %0d %h %h", nread, addr, got);
                    nread = nread + 1;
                    if (op == 8'h03 && got !== data) begin
                        mism = mism + 1;
                        if (mism <= 12)
                            $display("[MISMATCH] pc=%0d addr=%h got=%h expected=%h",
                                     pc - 1, addr, got, data);
                    end
                end

                8'h05: begin                                 // WAIT_RD_VALID
                    t = 0; st = 32'd0;
                    while (!st[3] && t < 1000) begin bus_read(16'h0001, st); t = t + 1; end
                    if (!st[3]) begin timeouts = timeouts + 1;
                        $display("[FAIL] pc=%0d rd_valid never set", pc - 1); end
                end

                8'h06: repeat (data) tick;

                8'hFF: running = 0;

                default: begin
                    $display("[FAIL] pc=%0d unknown opcode %h", pc - 1, op);
                    mism = mism + 1; running = 0;
                end
            endcase
        end

        $fclose(logfd);
`ifdef STROBES
        $display("STROBES acc_wr_en=%0d acc_rd_en=%0d", stb_wr, stb_rd);
`endif
        $display("PLAYER: %0d ops, %0d reads, %0d mismatches, %0d timeouts, %0d cycles polling",
                 nops, nread, mism, timeouts, cyc_total);
        if (mism == 0 && timeouts == 0 && axi_err == 0) $display("*** PLAYER PASSED ***");
        else                            $display("*** PLAYER FAILED ***");
        $finish;
    end

endmodule
