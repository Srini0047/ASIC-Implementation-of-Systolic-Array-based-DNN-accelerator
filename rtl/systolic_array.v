//=============================================================================
// Module      : systolic_array
// Description : ROWS x COLS grid of weight-stationary PEs (pure PE mesh).
//
//   Data movement (one PE hop per clock):
//     activations : left  -> right   (row r input = act_in[r], skewed outside)
//     partial sums: top   -> bottom  (top row psum_in tied to 0)
//     weights     : top   -> bottom  (shift-load, see below)
//
//   Weight loading: while weight_load=1 every PE's weight register shifts one
//   step down its column; column c is fed from wgt_in[c]. After ROWS load
//   cycles the word that was fed FIRST sits in the bottom row, so software
//   (the controller) must present weight-tile row ROWS-1 first, row 0 last.
//
//   Computes  psum_bottom[c] = sum_r  W[r][c] * act[r]   (one K=ROWS dot
//   product per output column, per activation vector).
//
//   Flattened buses: act_in[r*DW +: DW], wgt_in[c*DW +: DW],
//                    psum_out[c*ACC +: ACC].
//=============================================================================
module systolic_array #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,
    input  wire                         weight_load,
    input  wire [COLS*DATA_WIDTH-1:0]   wgt_in,     // fed into top of columns
    input  wire [ROWS*DATA_WIDTH-1:0]   act_in,     // already row-skewed
    output wire [COLS*ACC_WIDTH-1:0]    psum_out    // bottom-row partial sums
);

    // Inter-PE wiring, flattened.
    //   act_w  : index r*(COLS+1)+c  -> activation entering PE(r,c); c=COLS is
    //            the (unused) right-edge output.
    //   ps_w   : index r*COLS+c      -> psum entering PE(r,c); r=ROWS is the
    //            bottom-edge output.
    //   wt_w   : index r*COLS+c      -> weight entering PE(r,c); r=ROWS unused.
    // (Right-edge activation outputs and bottom-edge weight outputs are
    //  intentionally left unconnected -> UNUSEDSIGNAL waived.)
    /* verilator lint_off UNUSEDSIGNAL */
    wire [ROWS*(COLS+1)*DATA_WIDTH-1:0] act_w;
    wire [(ROWS+1)*COLS*ACC_WIDTH-1:0]  ps_w;
    wire [(ROWS+1)*COLS*DATA_WIDTH-1:0] wt_w;
    /* verilator lint_on UNUSEDSIGNAL */

    genvar r, c;
    generate
        // Left edge: activations from outside.
        for (r = 0; r < ROWS; r = r + 1) begin : g_left
            assign act_w[(r*(COLS+1))*DATA_WIDTH +: DATA_WIDTH] =
                   act_in[r*DATA_WIDTH +: DATA_WIDTH];
        end
        // Top edge: zero partial sum, weights from outside.
        for (c = 0; c < COLS; c = c + 1) begin : g_top
            assign ps_w[c*ACC_WIDTH +: ACC_WIDTH]   = {ACC_WIDTH{1'b0}};
            assign wt_w[c*DATA_WIDTH +: DATA_WIDTH] = wgt_in[c*DATA_WIDTH +: DATA_WIDTH];
        end

        for (r = 0; r < ROWS; r = r + 1) begin : g_row
            for (c = 0; c < COLS; c = c + 1) begin : g_col
                pe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .en          (en),
                    .weight_load (weight_load),
                    .weight_in   (wt_w [(r*COLS+c)*DATA_WIDTH +: DATA_WIDTH]),
                    .weight_out  (wt_w [((r+1)*COLS+c)*DATA_WIDTH +: DATA_WIDTH]),
                    .act_in      (act_w[(r*(COLS+1)+c)*DATA_WIDTH +: DATA_WIDTH]),
                    .act_out     (act_w[(r*(COLS+1)+c+1)*DATA_WIDTH +: DATA_WIDTH]),
                    .psum_in     (ps_w [(r*COLS+c)*ACC_WIDTH +: ACC_WIDTH]),
                    .psum_out    (ps_w [((r+1)*COLS+c)*ACC_WIDTH +: ACC_WIDTH])
                );
            end
        end

        // Bottom edge: partial sums leave the array.
        for (c = 0; c < COLS; c = c + 1) begin : g_bot
            assign psum_out[c*ACC_WIDTH +: ACC_WIDTH] =
                   ps_w[(ROWS*COLS+c)*ACC_WIDTH +: ACC_WIDTH];
        end
    endgenerate

endmodule
