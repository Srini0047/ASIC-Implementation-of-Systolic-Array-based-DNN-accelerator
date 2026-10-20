//=============================================================================
// Module      : dnn_accel_axil_top
// Description : SoC-ready wrapper: AXI4-Lite slave pins + interrupt.
//
//        CPU --AXI4-Lite--> axil_reg_bridge --reg bus--> dnn_accel_top
//
//   This is the module you would instantiate in an SoC / FPGA block design
//   (map it at any 256 KiB-aligned base address; the driver's word address x 4
//   is the byte offset). `irq` is a level output (done | err); it clears on the
//   next START.
//
//   Parameters are forwarded unchanged to dnn_accel_top (see that file).
//=============================================================================
module dnn_accel_axil_top #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter ACC_WIDTH  = 32,
    parameter ACT_DEPTH  = 256,
    parameter WGT_DEPTH  = 512,
    parameter BIAS_DEPTH = 16,
    parameter ACC_DEPTH  = 32,
    parameter LRB_DEPTH  = 8,
    parameter NUM_LAYERS = 4,
    parameter M_MAX      = 8,
    parameter KT_MAX     = 16,
    parameter NT_MAX     = 8,
    parameter PERF_EN    = 1
)(
    input  wire        aclk,
    input  wire        aresetn,          // asynchronous assert, synchronised inside

    input  wire [17:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,
    output wire        s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [17:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,
    output wire [1:0]  s_axil_rresp,
    output wire        s_axil_rvalid,
    input  wire        s_axil_rready,

    output wire        irq
);

    // The bridge shares the accelerator's synchronised reset so both come out of
    // reset together. rst_sync is instantiated here once and the accelerator's own
    // input synchroniser is disabled (SYNC_RST_IN = 0) to avoid a double chain.
    wire rst_n;
    rst_sync #(.STAGES(2)) u_rst_sync (.clk(aclk), .arst_n(aresetn), .srst_n(rst_n));

    wire        reg_wr_en, reg_rd_en;
    wire [15:0] reg_addr;
    wire [31:0] reg_wdata, reg_rdata;

    axil_reg_bridge u_bridge (
        .clk(aclk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),
        .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
        .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .reg_wr_en(reg_wr_en), .reg_rd_en(reg_rd_en), .reg_addr(reg_addr),
        .reg_wdata(reg_wdata), .reg_rdata(reg_rdata));

    dnn_accel_top #(
        .ROWS(ROWS), .COLS(COLS), .ACC_WIDTH(ACC_WIDTH), .ACT_DEPTH(ACT_DEPTH),
        .WGT_DEPTH(WGT_DEPTH), .BIAS_DEPTH(BIAS_DEPTH), .ACC_DEPTH(ACC_DEPTH),
        .LRB_DEPTH(LRB_DEPTH), .NUM_LAYERS(NUM_LAYERS), .M_MAX(M_MAX),
        .KT_MAX(KT_MAX), .NT_MAX(NT_MAX), .PERF_EN(PERF_EN),
        .SYNC_RST_IN(0)                        // reset already synchronised above
    ) u_acc (
        .clk(aclk), .rst_n_i(rst_n),
        .reg_wr_en(reg_wr_en), .reg_rd_en(reg_rd_en), .reg_addr(reg_addr),
        .reg_wdata(reg_wdata), .reg_rdata(reg_rdata), .irq(irq));

endmodule
