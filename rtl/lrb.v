//=============================================================================
// Module      : lrb  (Local Reuse Buffer)
// Description : Small on-chip buffer that keeps activation vectors close to
//               the array so they are NOT re-fetched from the big activation
//               SRAM for every weight tile.
//
//   Why it helps: computing C[M x (T*COLS)] = A[M x ROWS] * W[ROWS x (T*COLS)]
//   needs the same A vectors for each of the T column tiles. Tile 0 reads A
//   from the activation SRAM and, in the same pass, FILLS the LRB. Tiles
//   1..T-1 REPLAY vectors from the LRB. SRAM activation reads drop from T*M
//   to M + (T-1)*max(0, M-LRB_DEPTH). Weights are already reused M times
//   inside the PE weight registers (weight-stationary), so the LRB targets the
//   activation stream, the one operand that would otherwise be re-read.
//
//   Structure (all timing aligned to a 1-cycle read latency):
//       cycle c0   : controller issues SRAM read OR LRB read
//       cycle c0+1 : act_sram_data / lrb_rd_data valid; fill write happens
//                    here; source mux + output register capture the vector
//       cycle c0+2 : act_vec/act_vld valid at the array input
//
//   Controller-driven side-band signals (valid in cycle c0+1):
//       wr_en/wr_addr  : write act_sram_data into LRB slot (fill)
//       feed_vld       : a real vector is in flight
//       feed_sel_lrb   : 1 = take LRB data, 0 = take SRAM data
//   The output register forces act_vec to 0 when feed_vld=0 so the array is
//   flushed with zeros between tiles.
//=============================================================================
module lrb #(
    parameter ROWS       = 4,
    parameter DATA_WIDTH = 8,
    parameter LRB_DEPTH  = 8,         // number of activation vectors cached
    parameter AW         = 3          // address width, 2**AW >= LRB_DEPTH, AW>=1
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         en,

    // Data from the activation SRAM (registered read data).
    input  wire [ROWS*DATA_WIDTH-1:0]   act_sram_data,

    // Fill port (writes act_sram_data).
    input  wire                         wr_en,
    input  wire [AW-1:0]                wr_addr,

    // Replay read port (registered, 1-cycle latency).
    input  wire                         rd_en,
    input  wire [AW-1:0]                rd_addr,

    // Source select / valid, aligned with data (cycle c0+1).
    input  wire                         feed_vld,
    input  wire                         feed_sel_lrb,

    // To the compute core.
    output reg                          act_vld,
    output reg  [ROWS*DATA_WIDTH-1:0]   act_vec
);

    // Parameter sanity check (simulation only; ignored by synthesis).
    // synopsys translate_off
    initial if ((1 << AW) < LRB_DEPTH) begin
        $display("ERROR: lrb AW=%0d too small for LRB_DEPTH=%0d", AW, LRB_DEPTH);
        $finish;
    end
    // synopsys translate_on

    wire [ROWS*DATA_WIDTH-1:0] lrb_rd_data;

    sram_sdp #(
        .WIDTH (ROWS*DATA_WIDTH),
        .DEPTH (LRB_DEPTH),
        .AW    (AW)
    ) u_lrb_mem (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (wr_en),
        .wr_addr (wr_addr),
        .wr_data (act_sram_data),
        .rd_en   (rd_en),
        .rd_addr (rd_addr),
        .rd_data (lrb_rd_data)
    );

    // Source mux + output register (one pipeline stage keeps the mux off the
    // multiplier's critical path).
    always @(posedge clk) begin
        if (!rst_n) begin
            act_vld <= 1'b0;
            act_vec <= {ROWS*DATA_WIDTH{1'b0}};
        end else if (en) begin
            act_vld <= feed_vld;
            act_vec <= feed_vld ? (feed_sel_lrb ? lrb_rd_data : act_sram_data)
                                : {ROWS*DATA_WIDTH{1'b0}};
        end
    end

endmodule
