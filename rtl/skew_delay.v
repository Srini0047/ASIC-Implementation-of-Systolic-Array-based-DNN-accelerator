//=============================================================================
// Module      : skew_delay
// Description : Per-lane delay line used for
//                 - input skew   (DESKEW=0): lane i delayed by i cycles
//                 - output deskew(DESKEW=1): lane i delayed by LANES-1-i cycles
//
//   Systolic dataflow needs row r's activation to arrive r cycles after row 0
//   so that it meets the partial sum coming down from row r-1. Result columns
//   emerge column c at +c cycles, so they are re-aligned with the mirror delay.
//
//   Lane bus is flattened: lane i = bus[i*WIDTH +: WIDTH].
//   Zero-delay lanes are pure wires. Sync reset, shared enable.
//=============================================================================
module skew_delay #(
    parameter LANES  = 4,
    parameter WIDTH  = 8,
    parameter DESKEW = 0            // 0: delay = i, 1: delay = LANES-1-i
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     en,
    input  wire [LANES*WIDTH-1:0]   din,
    output wire [LANES*WIDTH-1:0]   dout
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : g_lane
            localparam integer D = (DESKEW == 0) ? i : (LANES - 1 - i);

            if (D == 0) begin : g_wire
                assign dout[i*WIDTH +: WIDTH] = din[i*WIDTH +: WIDTH];
            end else begin : g_sr
                // D-stage shift register, newest sample in the LSBs.
                reg [D*WIDTH-1:0] sr;
                if (D == 1) begin : g_d1
                    always @(posedge clk) begin
                        if (!rst_n)   sr <= {WIDTH{1'b0}};
                        else if (en)  sr <= din[i*WIDTH +: WIDTH];
                    end
                end else begin : g_dn
                    always @(posedge clk) begin
                        if (!rst_n)   sr <= {D*WIDTH{1'b0}};
                        else if (en)  sr <= {sr[(D-1)*WIDTH-1:0], din[i*WIDTH +: WIDTH]};
                    end
                end
                assign dout[i*WIDTH +: WIDTH] = sr[D*WIDTH-1 -: WIDTH];
            end
        end
    endgenerate

endmodule
