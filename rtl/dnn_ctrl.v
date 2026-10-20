// Lint waivers (intentional): narrow values are zero-extended into wider counters/addresses,
// and 16-bit descriptor / bus fields are truncated to the widths the hardware needs.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
//=============================================================================
// Module      : dnn_ctrl   (layer / tile sequencer)
// Description : Runs a whole network (up to NUM_LAYERS fully-connected layers)
//               from a table of layer descriptors, with no host involvement
//               between layers.
//
//   One layer computes   Y[M x N] = act( requant( X[M x K] * W[K x N] + b ) )
//   The K x N weight matrix is cut into ROWS x COLS tiles:
//        KT = ceil(K/ROWS) tiles along K,   NT = ceil(N/COLS) tiles along N.
//
//   Loop nest (this is the "dataflow schedule"):
//        for layer
//          for kt in 0..KT-1            <- outer: activation slice kt is reused
//            for nt in 0..NT-1          <- inner: ... by every N-tile (LRB!)
//              PASS(kt, nt):  LDW -> COMP -> DRAIN
//                acc[nt][m] (+)= X[m][slice kt] * W[tile kt,nt]      (all m)
//                if kt == KT-1:  Y[m][slice nt] = PPU(acc + bias)   -> act SRAM
//
//   Descriptor (16 x 16-bit registers per layer; fields 0..9 used):
//        0 M      batch rows in this layer            1..M_MAX
//        1 KT     K tiles                             1..KT_MAX
//        2 NT     N tiles                             1..NT_MAX
//        3 WBASE  weight-SRAM word address of tile (kt=0,nt=0)
//        4 BBASE  bias-SRAM word address of nt=0
//        5 SRC    act-SRAM word address of input  X  (word = X[m][slice kt] at
//                                                    SRC + m*KT + kt)
//        6 DST    act-SRAM word address of output Y  (word = Y[m][slice nt] at
//                                                    DST + m*NT + nt)
//        7 MULT   requant multiplier (unsigned 16 bit)
//        8 SHIFT  requant right shift (0..47)
//        9 RELU   1: clamp to [0,127], 0: clamp to [-128,127]
//   Weight tiles are stored in pass order p = kt*NT + nt, ROWS words per tile:
//        word address = WBASE + p*ROWS + r
//   Because Y is written in the same layout that the next layer reads as X
//   (needs ROWS == COLS), SRC/DST simply alternate between two regions of the
//   activation SRAM = ping-pong buffering.
//
//   Validation (FETCH state) rejects: zero or too-large M/KT/NT, M*NT larger
//   than the accumulator SRAM, SHIFT > 47. Address-range checking of the
//   *_BASE fields is the driver's job (sw/compile.py does it).
//
//   The FSM per pass is the same LDW/COMP/DRAIN sequence as accel_ctrl.v.
//   All memories still have 1-cycle registered reads.
//=============================================================================
module dnn_ctrl #(
    parameter ROWS       = 4,
    parameter ACC_DEPTH  = 32,
    parameter LRB_DEPTH  = 8,
    parameter NUM_LAYERS = 4,
    parameter M_MAX      = 8,
    parameter KT_MAX     = 16,
    parameter NT_MAX     = 8,
    // derived widths (computed by the parent)
    parameter M_W     = 4,   // clog2(M_MAX+1)
    parameter KT_W    = 5,   // clog2(KT_MAX+1)
    parameter NT_W    = 4,   // clog2(NT_MAX+1)
    parameter ACT_AW  = 8,
    parameter WGT_AW  = 9,
    parameter BIAS_AW = 4,
    parameter ACC_AW  = 5,
    parameter LRB_AW  = 3,
    parameter LAY_W   = 3,   // clog2(NUM_LAYERS+1)
    parameter DESC_AW = 6    // clog2(NUM_LAYERS*16)
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- host command ----
    input  wire                 start,              // 1-cycle pulse
    input  wire [LAY_W-1:0]     n_layers,           // 1..NUM_LAYERS
    output wire                 busy,
    output reg                  done,               // sticky until next start
    output reg                  err,                // sticky until next start

    // ---- descriptor write (16-bit registers, ignored while busy) ----
    input  wire                 desc_we,
    input  wire [DESC_AW-1:0]   desc_waddr,         // layer*16 + field
    input  wire [15:0]          desc_wdata,

    // ---- weight SRAM / PE array ----
    output wire                 wgt_rd_en,
    output wire [WGT_AW-1:0]    wgt_rd_addr,
    output reg                  wgt_load,

    // ---- activation SRAM / LRB ----
    output wire                 act_rd_en,
    output wire [ACT_AW-1:0]    act_rd_addr,
    output wire                 lrb_rd_en,
    output wire [LRB_AW-1:0]    lrb_rd_addr,
    output wire                 lrb_wr_en,
    output wire [LRB_AW-1:0]    lrb_wr_addr,
    output wire                 feed_vld,
    output wire                 feed_sel_lrb,

    // ---- bias SRAM ----
    output wire                 bias_rd_en,
    output wire [BIAS_AW-1:0]   bias_rd_addr,

    // ---- pass-level info for the datapath (accumulate + PPU) ----
    output wire                 pass_start,         // 1 cycle, first LDW cycle
    output wire [ACC_AW-1:0]    acc_base,           // accumulator row base (nt*M)
    output wire [ACT_AW-1:0]    dst_start,          // output pointer start
    output wire [NT_W-1:0]      dst_step,           // output pointer step (= NT)
    output wire                 first_kt,           // first K-tile: acc starts at 0
    output wire                 last_kt,            // last K-tile: run the PPU
    output wire [15:0]          ppu_mult,
    output wire [5:0]           ppu_shift,
    output wire                 ppu_relu,
    input  wire                 wr_done             // 1 pulse per finished vector
);

    //-------------------------------------------------------------------
    localparam integer LD_W  = (ROWS > 2) ? $clog2(ROWS) : 1;
    localparam [LD_W-1:0] LD_LAST = ROWS - 1;

    localparam [2:0] S_IDLE  = 3'd0,
                     S_FETCH = 3'd1,
                     S_LDW   = 3'd2,
                     S_COMP  = 3'd3,
                     S_DRAIN = 3'd4;

    localparam integer F_M = 0, F_KT = 1, F_NT = 2, F_WBASE = 3, F_BBASE = 4,
                       F_SRC = 5, F_DST = 6, F_MULT = 7, F_SHIFT = 8, F_RELU = 9;
    localparam integer SHIFT_MAX = 47;

    reg [2:0]        state;

    // descriptor storage (unused fields are pruned by synthesis)
    reg [15:0]       desc [0:NUM_LAYERS*16-1];
    always @(posedge clk)
        if (desc_we && !busy) desc[desc_waddr] <= desc_wdata;

    // per-layer latched configuration
    reg [LAY_W-1:0]  layer_cnt, n_lay_l;
    reg [M_W-1:0]    m_l;
    reg [KT_W-1:0]   kt_l;
    reg [NT_W-1:0]   nt_l;
    reg [ACT_AW-1:0] dst_base_l;
    reg [BIAS_AW-1:0] b_base_l;
    reg [15:0]       mult_l;
    reg [5:0]        shift_l;
    reg              relu_l;

    // loop counters and running pointers
    reg [KT_W-1:0]   kt_cnt;
    reg [NT_W-1:0]   nt_cnt;
    reg [M_W-1:0]    m_cnt;
    reg [LD_W-1:0]   ld_cnt;
    reg [WGT_AW-1:0] w_ptr;            // word address of current tile
    reg [ACT_AW-1:0] src_kt;           // SRC + kt
    reg [ACT_AW-1:0] act_addr;         // address of the vector being issued
    reg [ACC_AW-1:0] acc_base_r;       // nt * M
    reg [ACT_AW-1:0] dst_nt;           // DST + nt
    reg [BIAS_AW-1:0] b_nt;            // BBASE + nt
    reg [M_W-1:0]    done_cnt;         // vectors fully written this pass

    assign busy = (state != S_IDLE);

    //-------------------------------------------------------------------
    // Descriptor fetch + validation (comb, used in S_FETCH)
    //-------------------------------------------------------------------
    wire [15:0] f_m     = desc[layer_cnt*16 + F_M];
    wire [15:0] f_kt    = desc[layer_cnt*16 + F_KT];
    wire [15:0] f_nt    = desc[layer_cnt*16 + F_NT];
    wire [15:0] f_wbase = desc[layer_cnt*16 + F_WBASE];
    wire [15:0] f_bbase = desc[layer_cnt*16 + F_BBASE];
    wire [15:0] f_src   = desc[layer_cnt*16 + F_SRC];
    wire [15:0] f_dst   = desc[layer_cnt*16 + F_DST];
    wire [15:0] f_mult  = desc[layer_cnt*16 + F_MULT];
    wire [15:0] f_shift = desc[layer_cnt*16 + F_SHIFT];
    wire [15:0] f_relu  = desc[layer_cnt*16 + F_RELU];

    wire range_ok = (f_m  != 16'd0) && (f_m  <= M_MAX)  &&
                    (f_kt != 16'd0) && (f_kt <= KT_MAX) &&
                    (f_nt != 16'd0) && (f_nt <= NT_MAX) &&
                    (f_shift <= SHIFT_MAX);
    // Small multiplier on the *truncated* fields (only meaningful if range_ok).
    wire [M_W+NT_W-1:0] acc_rows = f_m[M_W-1:0] * f_nt[NT_W-1:0];
    wire fetch_ok = range_ok && (acc_rows <= ACC_DEPTH);

    //-------------------------------------------------------------------
    // Pass-level flags
    //-------------------------------------------------------------------
    wire kt_is_last = (kt_cnt == kt_l - 1'b1);
    wire nt_is_last = (nt_cnt == nt_l - 1'b1);
    assign first_kt   = (kt_cnt == {KT_W{1'b0}});
    assign last_kt    = kt_is_last;
    assign pass_start = (state == S_LDW) && (ld_cnt == {LD_W{1'b0}});
    assign acc_base   = acc_base_r;
    assign dst_start  = dst_nt;
    assign dst_step   = nt_l;
    assign ppu_mult   = mult_l;
    assign ppu_shift  = shift_l;
    assign ppu_relu   = relu_l;

    //-------------------------------------------------------------------
    // Weight read side (row ROWS-1 first), identical to accel_ctrl.v
    //-------------------------------------------------------------------
    localparam [WGT_AW:0] ROWS_M1 = ROWS - 1;
    wire [WGT_AW:0] ld_ext = {{(WGT_AW+1-LD_W){1'b0}}, ld_cnt};
    /* verilator lint_off UNUSEDSIGNAL */
    wire [WGT_AW:0] wgt_addr_w = {1'b0, w_ptr} + ROWS_M1 - ld_ext;
    /* verilator lint_on UNUSEDSIGNAL */
    assign wgt_rd_en   = (state == S_LDW);
    assign wgt_rd_addr = wgt_addr_w[WGT_AW-1:0];
    always @(posedge clk)
        if (!rst_n) wgt_load <= 1'b0;
        else        wgt_load <= wgt_rd_en;

    // Bias for this N-tile: one read at the start of the last-K pass; the SRAM
    // output register then holds it for the whole pass.
    assign bias_rd_en   = pass_start && kt_is_last;
    assign bias_rd_addr = b_nt;

    //-------------------------------------------------------------------
    // Activation issue side
    //-------------------------------------------------------------------
    wire iss_vld = (state == S_COMP);
    wire tile0   = (nt_cnt == {NT_W{1'b0}});          // first N-tile of this K slice
    wire in_lrb  = (m_cnt < LRB_DEPTH);
    wire multi   = (nt_l > 1);                        // reuse only pays if NT>1
    wire use_lrb = ~tile0 & in_lrb;                   // replay
    wire fill    =  tile0 & in_lrb & multi;           // fill while reading SRAM

    /* verilator lint_off UNUSEDSIGNAL */
    wire [M_W+LRB_AW-1:0] m_ext = {{LRB_AW{1'b0}}, m_cnt};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [LRB_AW-1:0]     idx   = m_ext[LRB_AW-1:0];

    assign act_rd_en   = iss_vld & ~use_lrb;
    assign act_rd_addr = act_addr;
    assign lrb_rd_en   = iss_vld &  use_lrb;
    assign lrb_rd_addr = idx;

    reg               feed_vld_r, feed_sel_r, fill_r;
    reg [LRB_AW-1:0]  idx_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            feed_vld_r <= 1'b0; feed_sel_r <= 1'b0; fill_r <= 1'b0;
            idx_r <= {LRB_AW{1'b0}};
        end else begin
            feed_vld_r <= iss_vld;
            feed_sel_r <= use_lrb;
            fill_r     <= iss_vld & fill;
            idx_r      <= idx;
        end
    end
    assign feed_vld     = feed_vld_r;
    assign feed_sel_lrb = feed_sel_r;
    assign lrb_wr_en    = fill_r;
    assign lrb_wr_addr  = idx_r;

    //-------------------------------------------------------------------
    // Main FSM
    //-------------------------------------------------------------------
    wire [ACT_AW-1:0]  kt_ext = kt_l;                 // zero-extend step
    wire [ACC_AW-1:0]  m_ext_acc = m_l;               // zero-extend step

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; err <= 1'b0;
            layer_cnt <= {LAY_W{1'b0}}; n_lay_l <= {LAY_W{1'b0}};
            m_l <= {M_W{1'b0}}; kt_l <= {KT_W{1'b0}}; nt_l <= {NT_W{1'b0}};
            dst_base_l <= {ACT_AW{1'b0}}; b_base_l <= {BIAS_AW{1'b0}};
            mult_l <= 16'd0; shift_l <= 6'd0; relu_l <= 1'b0;
            kt_cnt <= {KT_W{1'b0}}; nt_cnt <= {NT_W{1'b0}};
            m_cnt <= {M_W{1'b0}}; ld_cnt <= {LD_W{1'b0}};
            w_ptr <= {WGT_AW{1'b0}}; src_kt <= {ACT_AW{1'b0}};
            act_addr <= {ACT_AW{1'b0}}; acc_base_r <= {ACC_AW{1'b0}};
            dst_nt <= {ACT_AW{1'b0}}; b_nt <= {BIAS_AW{1'b0}};
            done_cnt <= {M_W{1'b0}};
        end else begin
            if (wr_done) done_cnt <= done_cnt + 1'b1;    // (DRAIN/FETCH clear below)

            case (state)
                //-----------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        done <= 1'b0;
                        if (n_layers != {LAY_W{1'b0}} && n_layers <= NUM_LAYERS) begin
                            err       <= 1'b0;
                            n_lay_l   <= n_layers;
                            layer_cnt <= {LAY_W{1'b0}};
                            state     <= S_FETCH;
                        end else begin
                            err <= 1'b1;
                        end
                    end
                end
                //-----------------------------------------------------
                S_FETCH: begin
                    if (fetch_ok) begin
                        m_l        <= f_m[M_W-1:0];
                        kt_l       <= f_kt[KT_W-1:0];
                        nt_l       <= f_nt[NT_W-1:0];
                        mult_l     <= f_mult;
                        shift_l    <= f_shift[5:0];
                        relu_l     <= f_relu[0];
                        dst_base_l <= f_dst[ACT_AW-1:0];
                        b_base_l   <= f_bbase[BIAS_AW-1:0];
                        w_ptr      <= f_wbase[WGT_AW-1:0];
                        src_kt     <= f_src[ACT_AW-1:0];
                        dst_nt     <= f_dst[ACT_AW-1:0];
                        b_nt       <= f_bbase[BIAS_AW-1:0];
                        acc_base_r <= {ACC_AW{1'b0}};
                        kt_cnt     <= {KT_W{1'b0}};
                        nt_cnt     <= {NT_W{1'b0}};
                        ld_cnt     <= {LD_W{1'b0}};
                        done_cnt   <= {M_W{1'b0}};
                        state      <= S_LDW;
                    end else begin
                        err   <= 1'b1;                    // abort the run
                        state <= S_IDLE;
                    end
                end
                //-----------------------------------------------------
                S_LDW: begin
                    if (ld_cnt == LD_LAST) begin
                        ld_cnt   <= {LD_W{1'b0}};
                        m_cnt    <= {M_W{1'b0}};
                        act_addr <= src_kt;
                        state    <= S_COMP;
                    end else begin
                        ld_cnt <= ld_cnt + 1'b1;
                    end
                end
                //-----------------------------------------------------
                S_COMP: begin
                    act_addr <= act_addr + kt_ext;        // next sample: +KT
                    if (m_cnt == m_l - 1'b1) begin
                        m_cnt <= {M_W{1'b0}};
                        state <= S_DRAIN;
                    end else begin
                        m_cnt <= m_cnt + 1'b1;
                    end
                end
                //-----------------------------------------------------
                S_DRAIN: begin
                    if (done_cnt == m_l) begin
                        done_cnt <= {M_W{1'b0}};
                        w_ptr    <= w_ptr + ROWS;         // next tile in pass order
                        if (!nt_is_last) begin
                            // next N-tile, same K slice
                            nt_cnt     <= nt_cnt + 1'b1;
                            acc_base_r <= acc_base_r + m_ext_acc;
                            dst_nt     <= dst_nt + 1'b1;
                            b_nt       <= b_nt + 1'b1;
                            state      <= S_LDW;
                        end else begin
                            nt_cnt     <= {NT_W{1'b0}};
                            acc_base_r <= {ACC_AW{1'b0}};
                            dst_nt     <= dst_base_l;
                            b_nt       <= b_base_l;
                            if (!kt_is_last) begin
                                kt_cnt <= kt_cnt + 1'b1;  // next K slice
                                src_kt <= src_kt + 1'b1;
                                state  <= S_LDW;
                            end else if (layer_cnt != n_lay_l - 1'b1) begin
                                layer_cnt <= layer_cnt + 1'b1;
                                state     <= S_FETCH;
                            end else begin
                                done  <= 1'b1;            // whole network finished
                                state <= S_IDLE;
                            end
                        end
                    end
                end
                //-----------------------------------------------------
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
