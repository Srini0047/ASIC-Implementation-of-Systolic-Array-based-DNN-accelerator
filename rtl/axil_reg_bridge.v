//=============================================================================
// Module      : axil_reg_bridge
// Description : AXI4-Lite SLAVE  ->  the accelerator's simple register bus.
//
//   Why this exists: a CPU (ARM/RISC-V) reaches peripherals through a bus
//   fabric, almost always AXI4-Lite for control registers. AXI-Lite has FIVE
//   independent channels, each with a VALID/READY handshake:
//
//        write address (AW)   write data (W)   write response (B)
//        read  address (AR)                    read  data     (R)
//
//   A transfer happens on a clock edge where VALID and READY are both high.
//   The accelerator's own bus (reg_wr_en/reg_rd_en/reg_addr/reg_wdata/reg_rdata)
//   is much simpler, so this block converts between the two, one transaction at
//   a time (no pipelining - control registers do not need throughput).
//
//   Address map: AXI byte address = 4 x register word address, i.e.
//   reg_addr = s_axil_awaddr[17:2] (a 256 KiB window; the map uses < 16 KiB).
//
//   Simplifications (documented so nobody is surprised):
//     * WSTRB is ignored: every write is a full 32-bit word (the driver only
//       does 32-bit stores).
//     * AWPROT/ARPROT ignored. Responses are always OKAY (2'b00); out-of-range
//       word addresses are ignored/zero-read inside dnn_reg_if.
//     * Write has priority over a simultaneous read request.
//
//   Timing of one transaction (cycle numbers are clock edges):
//     write : edge0 AW&W accepted -> edge1 reg_wr_en=1 & BVALID=1 -> B accepted
//     read  : edge0 AR accepted   -> edge1 reg_rd_en=1 -> edge2 RVALID=1,
//             RDATA = reg_rdata (dnn_reg_if registers its read data) -> R accepted
//
//   Single clock, synchronous reset (rst_n is already synchronised upstream by
//   dnn_accel_top's rst_sync; the bridge is reset with the same aresetn).
//=============================================================================
module axil_reg_bridge (
    input  wire        clk,
    input  wire        rst_n,

    // ---- AXI4-Lite slave ----
    input  wire [17:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,      // ignored (see header)
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,
    output reg         s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [17:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,
    output wire [1:0]  s_axil_rresp,
    output reg         s_axil_rvalid,
    input  wire        s_axil_rready,

    // ---- register bus (to dnn_accel_top) ----
    output reg         reg_wr_en,
    output reg         reg_rd_en,
    output reg  [15:0] reg_addr,
    output reg  [31:0] reg_wdata,
    input  wire [31:0] reg_rdata
);

    localparam [1:0] S_IDLE = 2'd0, S_B = 2'd1, S_R0 = 2'd2, S_R1 = 2'd3;
    reg [1:0] state;

    wire wr_req = s_axil_awvalid & s_axil_wvalid;                // both channels present
    wire idle   = (state == S_IDLE);

    // Ready may depend on valid (only VALID must never wait for READY).
    assign s_axil_awready = idle & wr_req;
    assign s_axil_wready  = idle & wr_req;
    assign s_axil_arready = idle & ~wr_req & s_axil_arvalid;

    assign s_axil_bresp = 2'b00;
    assign s_axil_rresp = 2'b00;
    assign s_axil_rdata = reg_rdata;          // held by dnn_reg_if until the next read

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            reg_wr_en     <= 1'b0;
            reg_rd_en     <= 1'b0;
            reg_addr      <= 16'd0;
            reg_wdata     <= 32'd0;
            s_axil_bvalid <= 1'b0;
            s_axil_rvalid <= 1'b0;
        end else begin
            reg_wr_en <= 1'b0;                // one-cycle strobes by default
            reg_rd_en <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (wr_req) begin
                        reg_addr      <= s_axil_awaddr[17:2];
                        reg_wdata     <= s_axil_wdata;
                        reg_wr_en     <= 1'b1;
                        s_axil_bvalid <= 1'b1;
                        state         <= S_B;
                    end else if (s_axil_arvalid) begin
                        reg_addr  <= s_axil_araddr[17:2];
                        reg_rd_en <= 1'b1;
                        state     <= S_R0;
                    end
                end
                S_B: if (s_axil_bready) begin
                    s_axil_bvalid <= 1'b0;
                    state         <= S_IDLE;
                end
                S_R0: begin                   // reg_rd_en is high this cycle
                    s_axil_rvalid <= 1'b1;    // reg_rdata is valid next cycle
                    state         <= S_R1;
                end
                S_R1: if (s_axil_rready) begin
                    s_axil_rvalid <= 1'b0;
                    state         <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // WSTRB and the byte-offset address bits are intentionally unused (word accesses only).
    wire _unused = &{1'b0, s_axil_wstrb, s_axil_awaddr[1:0], s_axil_araddr[1:0]};

endmodule
