//=============================================================================
// Module      : pe
// Description : Weight-stationary INT8 MAC processing element.
//
//   Refactored from the report's baseline PE.v. Changes vs. baseline:
//     1. SIGNED arithmetic. The baseline multiplied unsigned regs, which is
//        wrong for INT8 (negative weights/activations) and only passed the
//        report's tests because all stimulus was positive.
//     2. Synchronous active-low reset (baseline used async negedge rst_n).
//     3. Clock-enable `en` so the whole array can be idled (Genus can map the
//        enable to an ICG when clock gating is turned on).
//     4. `weight_out` port: the weight register doubles as the shift-chain
//        stage used to load weights down a column (no extra flops).
//     5. Product is sign-extended explicitly to ACC_WIDTH.
//
//   Dataflow (per clock, when en=1):
//     act_out  <= act_in                         (activation moves RIGHT)
//     psum_out <= psum_in + weight_reg * act_in  (partial sum moves DOWN)
//     if (weight_load) weight_reg <= weight_in   (weight shifts DOWN)
//
//   Overflow: ACC_WIDTH must be >= 2*DATA_WIDTH + clog2(ROWS). The default
//   (8/32) is safe for ROWS up to 2^16.
//=============================================================================
module pe #(
    parameter DATA_WIDTH = 8,     // INT8 operands
    parameter ACC_WIDTH  = 32     // wide accumulator, avoids overflow
)(
    input  wire                         clk,
    input  wire                         rst_n,        // sync, active low
    input  wire                         en,           // global clock enable
    input  wire                         weight_load,  // 1: shift weight in
    input  wire signed [DATA_WIDTH-1:0] weight_in,    // from PE above
    output wire signed [DATA_WIDTH-1:0] weight_out,   // to PE below
    input  wire signed [DATA_WIDTH-1:0] act_in,       // from PE on the left
    output reg  signed [DATA_WIDTH-1:0] act_out,      // to PE on the right
    input  wire signed [ACC_WIDTH-1:0]  psum_in,      // from PE above
    output reg  signed [ACC_WIDTH-1:0]  psum_out      // to PE below
);

    reg signed [DATA_WIDTH-1:0] weight_reg;

    // Signed multiply, then explicit sign extension to accumulator width.
    wire signed [2*DATA_WIDTH-1:0] product = weight_reg * act_in;
    /* verilator lint_off WIDTHEXPAND */
    wire signed [ACC_WIDTH-1:0]    product_ext = product;  // sign-extends
    /* verilator lint_on WIDTHEXPAND */

    assign weight_out = weight_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg <= {DATA_WIDTH{1'b0}};
            act_out    <= {DATA_WIDTH{1'b0}};
            psum_out   <= {ACC_WIDTH{1'b0}};
        end else if (en) begin
            if (weight_load)
                weight_reg <= weight_in;     // load / shift phase
            act_out  <= act_in;              // pass activation right
            psum_out <= psum_in + product_ext; // MAC, pass sum down
        end
    end

endmodule
