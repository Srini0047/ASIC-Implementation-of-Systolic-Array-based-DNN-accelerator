//=============================================================================
// Module      : sram_sdp
// Description : Behavioral simple-dual-port (1R1W) synchronous memory used
//               for the weight, activation and output buffers.
//
//   - Registered read: rd_data appears one cycle after rd_en/rd_addr.
//   - Read-during-write to the same address returns the OLD data.
//   - No reset on the array (standard for SRAM); rd_data is reset to 0.
//
//   ASIC NOTE: Genus will map this to flip-flops. For anything beyond a few
//   Kbit swap this wrapper's body for the foundry memory-compiler macro (keep
//   the port list) and add the macro's .lib/.lef to the flow.
//=============================================================================
module sram_sdp #(
    parameter WIDTH = 32,
    parameter DEPTH = 16,
    parameter AW    = 4              // must satisfy 2**AW >= DEPTH
)(
    input  wire                 clk,
    input  wire                 rst_n,
    // write port
    input  wire                 wr_en,
    input  wire [AW-1:0]        wr_addr,
    input  wire [WIDTH-1:0]     wr_data,
    // read port
    input  wire                 rd_en,
    input  wire [AW-1:0]        rd_addr,
    output reg  [WIDTH-1:0]     rd_data
);

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    always @(posedge clk) begin
        if (!rst_n)
            rd_data <= {WIDTH{1'b0}};
        else if (rd_en)
            rd_data <= mem[rd_addr];
    end

endmodule
