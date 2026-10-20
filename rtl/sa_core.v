//=============================================================================
// Module      : sa_core
// Description : Compute core = input skew + PE mesh + output deskew + valid
//               pipeline. Presents a simple "one vector in, one vector out"
//               streaming interface to the rest of the accelerator.
//
//   Timing (tau = cycle in which act_vec/act_vld are seen at the core input):
//     row r reaches PE(r,0) at tau+r,  PE(r,c) at tau+r+c
//     bottom psum of column c is valid at tau+ROWS+c
//     deskew delays column c by COLS-1-c   -> all columns valid at
//     tau + ROWS + COLS - 1  == tau + LATENCY
//   The valid bit rides an identical LATENCY-deep shift register.
//
//   Result vector m:  res_vec[c] = sum_r  W[r][c] * act_vec_m[r]  (INT32).
//   No back-pressure: results must be consumed the cycle res_vld is high.
//=============================================================================
module sa_core #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,      // sync, active low
    input  wire                         en,         // pipeline enable

    // Weight shift-load port (row ROWS-1 of the tile first).
    input  wire                         wgt_load,
    input  wire [COLS*DATA_WIDTH-1:0]   wgt_in,

    // Activation stream in (one K=ROWS vector per cycle).
    input  wire                         act_vld,
    input  wire [ROWS*DATA_WIDTH-1:0]   act_vec,

    // Result stream out (one COLS-wide vector per cycle).
    output wire                         res_vld,
    output wire                         res_vld_early,  // res_vld, one cycle ahead
    output wire [COLS*ACC_WIDTH-1:0]    res_vec
);

    localparam integer LATENCY = ROWS + COLS - 1;

    //-------------------------------------------------------------------
    // Input skew: row r delayed by r cycles.
    //-------------------------------------------------------------------
    wire [ROWS*DATA_WIDTH-1:0] act_skewed;

    skew_delay #(.LANES(ROWS), .WIDTH(DATA_WIDTH), .DESKEW(0)) u_in_skew (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (en),
        .din  (act_vec),
        .dout (act_skewed)
    );

    //-------------------------------------------------------------------
    // PE mesh.
    //-------------------------------------------------------------------
    wire [COLS*ACC_WIDTH-1:0] psum_raw;

    systolic_array #(
        .ROWS(ROWS), .COLS(COLS),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .en         (en),
        .weight_load(wgt_load),
        .wgt_in     (wgt_in),
        .act_in     (act_skewed),
        .psum_out   (psum_raw)
    );

    //-------------------------------------------------------------------
    // Output deskew: column c delayed by COLS-1-c cycles.
    //-------------------------------------------------------------------
    skew_delay #(.LANES(COLS), .WIDTH(ACC_WIDTH), .DESKEW(1)) u_out_deskew (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (en),
        .din  (psum_raw),
        .dout (res_vec)
    );

    //-------------------------------------------------------------------
    // Valid pipeline (LATENCY stages, aligned with res_vec).
    //-------------------------------------------------------------------
    reg [LATENCY-1:0] vld_pipe;
    always @(posedge clk) begin
        if (!rst_n)
            vld_pipe <= {LATENCY{1'b0}};
        else if (en)
            // Shift left by one, new valid enters at bit 0. Works for any
            // LATENCY >= 1 (no negative part-select when ROWS=COLS=1).
            vld_pipe <= (vld_pipe << 1) | {{(LATENCY-1){1'b0}}, act_vld};
    end
    // vfull[0] is the input valid, vfull[i] is the input valid delayed i cycles.
    // res_vld_early lets a consumer issue a memory read so that its data
    // arrives in the same cycle as res_vec (used for accumulate-in-memory).
    wire [LATENCY:0] vfull = {vld_pipe, act_vld};
    assign res_vld       = vfull[LATENCY];
    assign res_vld_early = vfull[LATENCY-1];

endmodule
