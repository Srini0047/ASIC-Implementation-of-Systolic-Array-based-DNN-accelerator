// Lint waivers (intentional): narrow values are zero-extended into wider counters/addresses,
// and 16-bit descriptor / bus fields are truncated to the widths the hardware needs.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
//=============================================================================
// Module      : dnn_reg_if   (32-bit register bus -> accelerator core)
// Description : Memory-mapped slave that a CPU / AXI-Lite bridge / testbench
//               uses to load weights, biases, layer descriptors and inputs,
//               start a run, poll status and read results back.
//
//   Bus: single-cycle write (reg_wr_en) or read (reg_rd_en). Reads return
//   reg_rdata one cycle later. No wait states, no bursts. An AXI4-Lite
//   wrapper only has to turn its handshakes into these two strobes.
//
//   Word addresses (32-bit words):
//     0x0000 CTRL        W  bit0 = START (write 1)
//     0x0001 STATUS      R  bit0 busy, bit1 done, bit2 err, bit3 rd_valid
//     0x0002 N_LAYERS    RW number of layers to run (1..NUM_LAYERS)
//     0x0003 ACT_RD_ADDR W  read act-SRAM word; poll rd_valid, then read data
//     0x0004 ID          R  0xD44A0100
//     0x0005 CFG_ARRAY   R  {ROWS, M_MAX, KT_MAX, NT_MAX}  (8 bits each)
//     0x0006 CFG_LAYERS  R  NUM_LAYERS
//     0x0007 CFG_ACT     R  ACT_DEPTH      0x0008 CFG_WGT   R  WGT_DEPTH
//     0x0009 CFG_BIAS    R  BIAS_DEPTH     0x000A CFG_ACC   R  ACC_DEPTH
//     0x000B CFG_LRB     R  LRB_DEPTH
//     0x0010+b ACT_RD_DATA[b]  R  beat b of the act word read via ACT_RD_ADDR
//     0x0020 PERF_CYCLES 0x0021 PERF_ACT_RD 0x0022 PERF_LRB_RD
//     0x0023 PERF_WGT_RD 0x0024 PERF_VEC                          (R)
//     0x0100 + layer*16 + field   layer descriptor (W, low 16 bits)
//     0x1000 + word*BSTRIDE_W + beat   weight SRAM window   (W)
//     0x2000 + word*BSTRIDE_A + beat   activation SRAM window (W)
//     0x3000 + word*BSTRIDE_B + beat   bias SRAM window (W, one INT32 per beat)
//
//   Wide memory words are written as 32-bit beats, LSB beat first; the write
//   of the LAST beat commits the word. BSTRIDE_x is BEATS_x rounded up to a
//   power of two (BEATS_W = ceil(8*COLS/32), BEATS_A = ceil(8*ROWS/32),
//   BEATS_B = COLS). Lower beats are held in a staging register.
//
//   Race-free start: STATUS reports busy=1/done=0 from the cycle after the
//   START write, before the core FSM has moved (start_r covers the gap).
//=============================================================================
module dnn_reg_if #(
    parameter ROWS       = 4,
    parameter COLS       = 4,
    parameter ACC_WIDTH  = 32,       // <= 32 (one lane per bias beat)
    parameter ACT_DEPTH  = 256,
    parameter WGT_DEPTH  = 512,
    parameter BIAS_DEPTH = 16,
    parameter ACC_DEPTH  = 32,
    parameter LRB_DEPTH  = 8,
    parameter NUM_LAYERS = 4,
    parameter M_MAX      = 8,
    parameter KT_MAX     = 16,
    parameter NT_MAX     = 8,
    parameter ACT_AW     = 8,
    parameter WGT_AW     = 9,
    parameter BIAS_AW    = 4,
    parameter LAY_W      = 3,
    parameter DESC_AW    = 6
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // register bus
    input  wire                      reg_wr_en,
    input  wire                      reg_rd_en,
    input  wire [15:0]               reg_addr,
    input  wire [31:0]               reg_wdata,
    output reg  [31:0]               reg_rdata,

    // to core
    output reg                       start,
    output reg  [LAY_W-1:0]          n_layers,
    input  wire                      busy,
    input  wire                      done,
    input  wire                      err,
    output reg                       desc_we,
    output reg  [DESC_AW-1:0]        desc_waddr,
    output reg  [15:0]               desc_wdata,
    output reg                       wgt_we,
    output reg  [WGT_AW-1:0]         wgt_waddr,
    output reg  [COLS*8-1:0]         wgt_wdata,
    output reg                       bias_we,
    output reg  [BIAS_AW-1:0]        bias_waddr,
    output reg  [COLS*ACC_WIDTH-1:0] bias_wdata,
    output reg                       act_we,
    output reg  [ACT_AW-1:0]         act_waddr,
    output reg  [ROWS*8-1:0]         act_wdata,
    output reg                       act_re,
    output reg  [ACT_AW-1:0]         act_raddr,
    input  wire [ROWS*8-1:0]         act_rdata,
    input  wire [31:0]               perf_cycles,
    input  wire [31:0]               perf_act_rd,
    input  wire [31:0]               perf_lrb_rd,
    input  wire [31:0]               perf_wgt_rd,
    input  wire [31:0]               perf_vec
);

    //-------------------------------------------------------------------
    // Beat geometry
    //-------------------------------------------------------------------
    localparam integer BEATS_W = (COLS*8 + 31) / 32;
    localparam integer BEATS_A = (ROWS*8 + 31) / 32;
    localparam integer BEATS_B = COLS;
    localparam integer SH_W = (BEATS_W > 1) ? $clog2(BEATS_W) : 0;
    localparam integer SH_A = (BEATS_A > 1) ? $clog2(BEATS_A) : 0;
    localparam integer SH_B = (BEATS_B > 1) ? $clog2(BEATS_B) : 0;
    localparam integer MAXB = COLS;               // >= all three

    // 8-bit copies of the configuration parameters for the CFG_ARRAY register
    // (a part-select of a parameter is not portable Verilog-2001).
    localparam [7:0] R8 = ROWS, M8 = M_MAX, KT8 = KT_MAX, NT8 = NT_MAX;

    //-------------------------------------------------------------------
    // Address decode
    //-------------------------------------------------------------------
    wire [3:0]  region = reg_addr[15:12];
    wire [11:0] offs   = reg_addr[11:0];
    wire is_ctrl  = (reg_addr == 16'h0000);
    wire is_nlay  = (reg_addr == 16'h0002);
    wire is_ardad = (reg_addr == 16'h0003);
    wire is_desc  = (reg_addr[15:8] == 8'h01);
    wire is_wgt   = (region == 4'h1);
    wire is_act   = (region == 4'h2);
    wire is_bias  = (region == 4'h3);

    // window word index / beat
    wire [11:0] widx_w = offs >> SH_W;
    wire [11:0] widx_a = offs >> SH_A;
    wire [11:0] widx_b = offs >> SH_B;
    wire [11:0] beat_w = offs & ((12'd1 << SH_W) - 12'd1);
    wire [11:0] beat_a = offs & ((12'd1 << SH_A) - 12'd1);
    wire [11:0] beat_b = offs & ((12'd1 << SH_B) - 12'd1);
    wire last_w = (beat_w == BEATS_W - 1);
    wire last_a = (beat_a == BEATS_A - 1);
    wire last_b = (beat_b == BEATS_B - 1);

    wire [11:0] beat_sel = is_wgt ? beat_w : is_act ? beat_a : beat_b;

    // staging register (shared: the host writes one window word at a time)
    reg  [MAXB*32-1:0] stage;
    reg  [MAXB*32-1:0] asm;
    always @* begin
        asm = stage;
        asm[beat_sel[3:0]*32 +: 32] = reg_wdata;    // beat index < 16
    end

    reg start_r;                                     // START seen last cycle

    //-------------------------------------------------------------------
    // Writes (all outputs registered = clean timing to the core)
    //-------------------------------------------------------------------
    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            start <= 1'b0; start_r <= 1'b0; n_layers <= {LAY_W{1'b0}};
            desc_we <= 1'b0; wgt_we <= 1'b0; bias_we <= 1'b0; act_we <= 1'b0;
            act_re <= 1'b0; stage <= {MAXB*32{1'b0}};
            desc_waddr <= {DESC_AW{1'b0}}; desc_wdata <= 16'd0;
            wgt_waddr <= {WGT_AW{1'b0}}; wgt_wdata <= {COLS*8{1'b0}};
            bias_waddr <= {BIAS_AW{1'b0}}; bias_wdata <= {COLS*ACC_WIDTH{1'b0}};
            act_waddr <= {ACT_AW{1'b0}}; act_wdata <= {ROWS*8{1'b0}};
            act_raddr <= {ACT_AW{1'b0}};
        end else begin
            // strobes default low
            start <= 1'b0; desc_we <= 1'b0; wgt_we <= 1'b0;
            bias_we <= 1'b0; act_we <= 1'b0; act_re <= 1'b0;
            start_r <= 1'b0;

            if (reg_wr_en) begin
                if (is_ctrl && reg_wdata[0]) begin
                    start <= 1'b1; start_r <= 1'b1;
                end
                // An out-of-range layer count must not wrap into a legal one (e.g. 9 -> 1 with
                // 3 bits): store 0, which the controller rejects with err on START.
                if (is_nlay)  n_layers <= (reg_wdata > NUM_LAYERS) ? {LAY_W{1'b0}} : reg_wdata[LAY_W-1:0];
                if (is_ardad) begin
                    act_raddr <= reg_wdata[ACT_AW-1:0]; act_re <= 1'b1;
                end
                if (is_desc && reg_addr[7:0] < NUM_LAYERS*16) begin
                    desc_we <= 1'b1; desc_waddr <= reg_addr[DESC_AW-1:0];
                    desc_wdata <= reg_wdata[15:0];
                end
                if (is_wgt) begin
                    if (last_w) begin
                        if (widx_w < WGT_DEPTH) begin
                            wgt_we <= 1'b1; wgt_waddr <= widx_w[WGT_AW-1:0];
                            wgt_wdata <= asm[COLS*8-1:0];
                        end
                    end else stage[beat_w[3:0]*32 +: 32] <= reg_wdata;
                end
                if (is_act) begin
                    if (last_a) begin
                        if (widx_a < ACT_DEPTH) begin
                            act_we <= 1'b1; act_waddr <= widx_a[ACT_AW-1:0];
                            act_wdata <= asm[ROWS*8-1:0];
                        end
                    end else stage[beat_a[3:0]*32 +: 32] <= reg_wdata;
                end
                if (is_bias) begin
                    if (last_b) begin
                        if (widx_b < BIAS_DEPTH) begin
                            bias_we <= 1'b1; bias_waddr <= widx_b[BIAS_AW-1:0];
                            for (k = 0; k < COLS; k = k + 1)
                                bias_wdata[k*ACC_WIDTH +: ACC_WIDTH] <= asm[k*32 +: ACC_WIDTH];
                        end
                    end else stage[beat_b[3:0]*32 +: 32] <= reg_wdata;
                end
            end
        end
    end

    //-------------------------------------------------------------------
    // Reads (registered, 1-cycle latency)
    //-------------------------------------------------------------------
    wire [31:0] status = {28'd0,
                          ~act_re,                    // rd_valid (0 while a read is in flight)
                          err & ~start_r,             // masked like done: never show a stale error after START
                          done & ~start_r,
                          busy | start_r};

    wire [7:0] a_beat = reg_addr[7:0] - 8'h10;        // valid for 0x10..0x17
    always @(posedge clk) begin
        if (!rst_n) reg_rdata <= 32'd0;
        else if (reg_rd_en) begin
            reg_rdata <= 32'd0;
            case (reg_addr)
                16'h0001: reg_rdata <= status;
                16'h0002: reg_rdata <= {{(32-LAY_W){1'b0}}, n_layers};
                16'h0004: reg_rdata <= 32'hD44A0100;
                16'h0005: reg_rdata <= {R8, M8, KT8, NT8};
                16'h0006: reg_rdata <= NUM_LAYERS;
                16'h0007: reg_rdata <= ACT_DEPTH;
                16'h0008: reg_rdata <= WGT_DEPTH;
                16'h0009: reg_rdata <= BIAS_DEPTH;
                16'h000A: reg_rdata <= ACC_DEPTH;
                16'h000B: reg_rdata <= LRB_DEPTH;
                16'h0020: reg_rdata <= perf_cycles;
                16'h0021: reg_rdata <= perf_act_rd;
                16'h0022: reg_rdata <= perf_lrb_rd;
                16'h0023: reg_rdata <= perf_wgt_rd;
                16'h0024: reg_rdata <= perf_vec;
                default: begin
                    if (reg_addr[15:3] == 13'h0002)   // 0x0010..0x0017
                        reg_rdata <= act_rdata >> (a_beat[2:0] * 32);
                end
            endcase
        end
    end

endmodule
