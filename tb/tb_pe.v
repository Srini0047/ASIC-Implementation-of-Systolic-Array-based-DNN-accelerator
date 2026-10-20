//=============================================================================
// Testbench   : tb_pe
// Description : Self-checking upgrade of the report's pe_tb. Keeps the
//               report's four original vectors (5x3+0, 5x4+10, 5x7+20) and adds
//               signed corners (negative weights/activations, -128 x -128),
//               weight hold during compute, weight shift-through (weight_out),
//               enable-gating and synchronous reset.
//   Run: iverilog -g2005 -o pe_sim tb/tb_pe.v rtl/pe.v && vvp pe_sim
//=============================================================================
`timescale 1ns/1ps
module tb_pe;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;

    reg                          clk = 0;
    reg                          rst_n, en, weight_load;
    reg  signed [DATA_WIDTH-1:0] weight_in, act_in;
    reg  signed [ACC_WIDTH-1:0]  psum_in;
    wire signed [DATA_WIDTH-1:0] weight_out, act_out;
    wire signed [ACC_WIDTH-1:0]  psum_out;
    integer errors = 0;

    pe #(.DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .weight_load(weight_load),
        .weight_in(weight_in), .weight_out(weight_out),
        .act_in(act_in), .act_out(act_out),
        .psum_in(psum_in), .psum_out(psum_out));

    always #5 clk = ~clk;

    // Apply one cycle of inputs, then check outputs after the next clock edge.
    task step;
        input signed [DATA_WIDTH-1:0] a;
        input signed [ACC_WIDTH-1:0]  p;
        input signed [ACC_WIDTH-1:0]  exp_psum;
        begin
            act_in = a; psum_in = p;
            @(posedge clk); #1;
            if (psum_out !== exp_psum || act_out !== a) begin
                $display("[FAIL] act=%0d psum_in=%0d -> psum_out=%0d (exp %0d) act_out=%0d",
                         a, p, psum_out, exp_psum, act_out);
                errors = errors + 1;
            end else
                $display("[ok]   act=%0d psum_in=%0d -> psum_out=%0d act_out=%0d", a, p, psum_out, act_out);
        end
    endtask

    task load_w;
        input signed [DATA_WIDTH-1:0] w;
        begin
            weight_in = w; weight_load = 1; act_in = 0; psum_in = 0;
            @(posedge clk); #1; weight_load = 0;
            if (weight_out !== w) begin
                $display("[FAIL] weight_out=%0d exp %0d", weight_out, w); errors = errors + 1; end
        end
    endtask

    initial begin
        rst_n = 0; en = 1; weight_load = 0; weight_in = 0; act_in = 0; psum_in = 0;
        repeat (2) @(posedge clk); #1; rst_n = 1;

        // ---- report's original vectors (weight = 5) ----
        load_w(5);
        step(3, 0, 15);
        step(4, 10, 30);
        step(7, 20, 55);

        // ---- signed corners ----
        load_w(-3);   step(7, 0, -21);  step(-7, 5, 26);
        load_w(-128); step(-128, 0, 16384); step(127, 0, -16256); step(-128, -16384, 0);
        load_w(127);  step(127, 0, 16129);  step(-1, 100, -27);
        // accumulator width: large positive partial sum + product
        step(127, 32'sh7000_0000, 32'sh7000_0000 + 16129);

        // ---- weight holds when weight_load=0 even if weight_in changes ----
        weight_in = 8'sd9; step(2, 0, 254);

        // ---- enable gating: outputs freeze when en=0 ----
        en = 0; act_in = 55; psum_in = 999; repeat (3) @(posedge clk); #1;
        if (psum_out !== 254 || act_out !== 2) begin
            $display("[FAIL] outputs moved while en=0"); errors = errors + 1; end
        else $display("[ok]   en=0 freezes state");
        en = 1;

        // ---- synchronous reset clears weight and outputs ----
        rst_n = 0; @(posedge clk); #1; rst_n = 1;
        if (psum_out !== 0 || act_out !== 0 || weight_out !== 0) begin
            $display("[FAIL] reset did not clear"); errors = errors + 1; end
        else $display("[ok]   sync reset");
        step(100, 7, 7);          // weight is 0 after reset: psum passes through

        if (errors == 0) $display("*** tb_pe PASSED ***");
        else             $display("*** tb_pe FAILED: %0d error(s) ***", errors);
        $finish;
    end
endmodule
