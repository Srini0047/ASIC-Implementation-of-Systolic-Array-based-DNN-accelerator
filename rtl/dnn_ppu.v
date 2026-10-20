// Lint waivers (intentional): narrow values are zero-extended into wider counters/addresses,
// and 16-bit descriptor / bus fields are truncated to the widths the hardware needs.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
//=============================================================================
// Module      : dnn_ppu   (Post-Processing Unit / "epilogue")
// Description : Turns the INT32 accumulator results of a layer into the INT8
//               activations the NEXT layer consumes:
//
//                  y = clamp( ( (acc + bias) * mult + round ) >>> shift )
//
//               - bias   : INT32 per output feature
//               - mult   : unsigned MULT_W-bit fixed-point multiplier
//               - shift  : right shift, 0 .. SHIFT_MAX
//               - round  : 1 << (shift-1)   (round half up), 0 if shift==0
//               - clamp  : [-128,127], or [0,127] when relu=1 (ReLU folded in)
//
//   Together, mult/2^shift approximates the real-valued rescale
//   (s_input * s_weight) / s_output produced by the quantizer (sw/quantize.py).
//   This is the standard "integer-only inference" requantization
//   (Jacob et al., CVPR 2018 - the report's reference [4]).
//
//   Three pipeline stages (1 lane-parallel add, 1 multiply, 1 round/shift/clamp)
//   so the 49-bit multiply is not chained with the adder or the clamp.
//   Latency: PPU_LAT = 3 cycles from in_vld to out_vld.
//
//   The arithmetic is bit-exact against sw/golden.py (int64 numpy).
//=============================================================================
module dnn_ppu #(
    parameter LANES     = 4,
    parameter ACC_WIDTH = 32,
    parameter MULT_W    = 16,
    parameter SHIFT_W   = 6
)(
    input  wire                         clk,
    input  wire                         rst_n,      // sync, active low
    input  wire                         en,

    input  wire                         in_vld,
    input  wire [LANES*ACC_WIDTH-1:0]   in_data,    // accumulated sums
    input  wire [LANES*ACC_WIDTH-1:0]   bias,       // per-lane INT32 bias

    input  wire [MULT_W-1:0]            mult,       // unsigned
    input  wire [SHIFT_W-1:0]           shift,
    input  wire                         relu,

    output reg                          out_vld,
    output wire [LANES*8-1:0]           out_data    // INT8 per lane
);

    localparam integer SW = ACC_WIDTH + 1;              // sum width (no overflow)
    localparam integer PW = SW + MULT_W + 1;            // product width

    // Valid pipeline (3 stages).
    reg v1, v2;
    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0; v2 <= 1'b0; out_vld <= 1'b0;
        end else if (en) begin
            v1 <= in_vld; v2 <= v1; out_vld <= v2;
        end
    end

    // Rounding constant: 1 << (shift-1), or 0 when shift == 0.
    wire signed [PW-1:0] rnd =
        (shift == {SHIFT_W{1'b0}}) ? {PW{1'b0}}
                                   : ({{(PW-1){1'b0}}, 1'b1} << (shift - 1'b1));

    // Both branches are signed literals, so -128 is sign-extended correctly
    // (mixing in an unsigned concatenation would silently make it unsigned).
    wire signed [PW-1:0] clamp_hi = 127;
    wire signed [PW-1:0] clamp_lo = relu ? 0 : -128;
    wire signed [MULT_W:0] mult_s = {1'b0, mult};       // force positive

    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : g_lane
            reg signed [SW-1:0] sum_r;                  // stage 1
            reg signed [PW-1:0] prod_r;                 // stage 2
            reg        [7:0]    q_r;                    // stage 3

            wire signed [ACC_WIDTH-1:0] a = in_data[l*ACC_WIDTH +: ACC_WIDTH];
            wire signed [ACC_WIDTH-1:0] b = bias   [l*ACC_WIDTH +: ACC_WIDTH];

            // stage 3 combinational: round, arithmetic shift, clamp
            wire signed [PW-1:0] shifted = (prod_r + rnd) >>> shift;
            wire signed [PW-1:0] limited =
                (shifted > clamp_hi) ? clamp_hi :
                (shifted < clamp_lo) ? clamp_lo : shifted;

            always @(posedge clk) begin
                if (!rst_n) begin
                    sum_r  <= {SW{1'b0}};
                    prod_r <= {PW{1'b0}};
                    q_r    <= 8'd0;
                end else if (en) begin
                    sum_r  <= a + b;                    // 1: bias add (sign-extends)
                    prod_r <= sum_r * mult_s;           // 2: scale multiply
                    q_r    <= limited[7:0];             // 3: round/shift/clamp
                end
            end

            assign out_data[l*8 +: 8] = q_r;
        end
    endgenerate

endmodule
