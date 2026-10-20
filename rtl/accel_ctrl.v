//=============================================================================
// Module      : accel_ctrl
// Description : Sequencer for a tiled matrix multiply
//                   C[M x (T*COLS)] = A[M x ROWS] * W[ROWS x (T*COLS)]
//               INT8 in, INT32 out. One weight tile (ROWS x COLS) is resident
//               in the PEs at a time (weight-stationary); the M activation
//               vectors are streamed through it; then the next tile is loaded.
//
//   FSM (one-hot friendly 2-bit encoding, Genus will re-encode if asked):
//
//        start & cfg_ok
//   IDLE ----------------> LDW ---------------> COMP ---------> DRAIN
//    ^                      ^   ROWS cycles       M cycles         |
//    |                      |                                      |
//    |                      +------- more tiles (res_cnt==M) ------+
//    +------------------------- last tile, set done ---------------+
//
//   IDLE : wait for `start`. Config validated combinationally (cfg_ok);
//          a bad config sets cfg_err and the FSM stays idle.
//   LDW  : issue ROWS weight-SRAM reads, tile row ROWS-1 first. wgt_load (the
//          registered read-enable) shifts the words down the PE columns.
//   COMP : issue M activation reads. Tile 0 -> activation SRAM (+ fill LRB
//          for vectors m < LRB_DEPTH). Tiles >=1 -> LRB when m < LRB_DEPTH,
//          otherwise SRAM.
//   DRAIN: wait until all M result vectors of this tile have been written to
//          the output SRAM (res_cnt == M). Zeros flush the mesh meanwhile, so
//          the next tile's weight load starts on a clean pipeline.
//
//   Output SRAM layout: tile-major, row address = tile*M + m (achieved with a
//   running write pointer, no multiplier).
//   Weight SRAM layout: address = tile*ROWS + r  (word = COLS INT8 weights).
//   Activation SRAM   : address = m             (word = ROWS INT8 values).
//
//   Read latency of every memory is 1 cycle; this is why the LRB fill and
//   source-select side-band signals are delayed one cycle (feed_*_r regs).
//=============================================================================
module accel_ctrl #(
    parameter ROWS       = 4,
    parameter ACT_DEPTH  = 16,
    parameter WGT_DEPTH  = 16,
    parameter OUT_DEPTH  = 32,
    parameter LRB_DEPTH  = 8,
    // Derived widths (computed by the parent; ports cannot use body localparams)
    parameter M_W        = 5,    // clog2(ACT_DEPTH+1)
    parameter T_W        = 3,    // clog2(WGT_DEPTH/ROWS + 1)
    parameter ACT_AW     = 4,    // clog2(ACT_DEPTH), >=1
    parameter WGT_AW     = 4,    // clog2(WGT_DEPTH), >=1
    parameter OUT_AW     = 5,    // clog2(OUT_DEPTH), >=1
    parameter LRB_AW     = 3     // clog2(LRB_DEPTH), >=1
)(
    input  wire                 clk,
    input  wire                 rst_n,           // sync, active low

    // Host control / status
    input  wire                 start,           // pulse; ignored when busy
    input  wire [M_W-1:0]       cfg_m,           // # activation vectors (rows of A)
    input  wire [T_W-1:0]       cfg_tiles,       // # weight column-tiles
    output wire                 busy,
    output reg                  done,            // sticky, cleared by next start
    output reg                  cfg_err,         // sticky, cleared by next good start

    // Weight SRAM read + PE weight shift enable
    output wire                 wgt_rd_en,
    output wire [WGT_AW-1:0]    wgt_rd_addr,
    output reg                  wgt_load,        // aligned with SRAM rd data

    // Activation SRAM read
    output wire                 act_rd_en,
    output wire [ACT_AW-1:0]    act_rd_addr,

    // LRB control
    output wire                 lrb_rd_en,
    output wire [LRB_AW-1:0]    lrb_rd_addr,
    output wire                 lrb_wr_en,       // aligned with SRAM rd data
    output wire [LRB_AW-1:0]    lrb_wr_addr,
    output wire                 feed_vld,        // aligned with SRAM rd data
    output wire                 feed_sel_lrb,    // aligned with SRAM rd data

    // Result stream from compute core -> output SRAM write
    input  wire                 res_vld,
    output wire                 out_wr_en,
    output wire [OUT_AW-1:0]    out_wr_addr
);

    //-------------------------------------------------------------------
    // Local helpers / constants
    //-------------------------------------------------------------------
    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction

    localparam integer MAX_TILES = WGT_DEPTH / ROWS;
    localparam integer LD_W      = (clog2(ROWS) < 1) ? 1 : clog2(ROWS);

    localparam [1:0] S_IDLE  = 2'd0,
                     S_LDW   = 2'd1,
                     S_COMP  = 2'd2,
                     S_DRAIN = 2'd3;

    reg [1:0]        state;
    reg [T_W-1:0]    tile_cnt;    // current tile
    reg [T_W-1:0]    t_lat;       // latched cfg_tiles
    reg [M_W-1:0]    m_lat;       // latched cfg_m
    reg [M_W-1:0]    m_cnt;       // activation vector index issued in COMP
    reg [LD_W-1:0]   ld_cnt;      // weight row counter in LDW
    reg [WGT_AW-1:0] wgt_base;    // tile*ROWS, updated by addition
    reg [M_W-1:0]    res_cnt;     // results received for the current tile
    reg [OUT_AW-1:0] out_wptr;    // running output-SRAM write pointer

    //-------------------------------------------------------------------
    // Config validation (static inputs; not timing critical)
    //-------------------------------------------------------------------
    // Typed constants (each provably fits its width: M_W/T_W are derived
    // from ACT_DEPTH / MAX_TILES by the parent).
    /* verilator lint_off WIDTHTRUNC */
    localparam [M_W-1:0]    ACT_DEPTH_C = ACT_DEPTH;
    localparam [T_W-1:0]    MAX_TILES_C = MAX_TILES;
    localparam [LD_W-1:0]   LD_LAST     = ROWS - 1;
    /* verilator lint_on WIDTHTRUNC */

    wire [M_W+T_W-1:0] total_rows = cfg_m * cfg_tiles;
    /* verilator lint_off WIDTHEXPAND */
    wire cfg_ok = (cfg_m != {M_W{1'b0}}) && (cfg_m <= ACT_DEPTH_C) &&
                  (cfg_tiles != {T_W{1'b0}}) && (cfg_tiles <= MAX_TILES_C) &&
                  (total_rows <= OUT_DEPTH);
    /* verilator lint_on WIDTHEXPAND */

    assign busy = (state != S_IDLE);

    //-------------------------------------------------------------------
    // Weight read side. Row ROWS-1 first so it ends at the bottom PE row.
    //-------------------------------------------------------------------
    localparam [WGT_AW:0] ROWS_M1 = ROWS - 1;
    wire [WGT_AW:0] ld_ext     = {{(WGT_AW+1-LD_W){1'b0}}, ld_cnt};
    /* verilator lint_off UNUSEDSIGNAL */
    wire [WGT_AW:0] wgt_addr_w = {1'b0, wgt_base} + ROWS_M1 - ld_ext;
    /* verilator lint_on UNUSEDSIGNAL */
    assign wgt_rd_en   = (state == S_LDW);
    assign wgt_rd_addr = wgt_addr_w[WGT_AW-1:0];   // MSB never needed

    always @(posedge clk) begin
        if (!rst_n) wgt_load <= 1'b0;
        else        wgt_load <= wgt_rd_en;      // SRAM data valid next cycle
    end

    //-------------------------------------------------------------------
    // Activation issue side (COMP state)
    //-------------------------------------------------------------------
    wire iss_vld = (state == S_COMP);
    wire tile0   = (tile_cnt == {T_W{1'b0}});
    wire in_lrb  = (m_cnt < LRB_DEPTH);           // vector index fits in LRB
    wire multi   = (t_lat > 1);                   // reuse only pays off if >1 tile
    wire use_lrb = ~tile0 & in_lrb;               // replay from LRB
    wire fill    =  tile0 & in_lrb & multi;       // fill LRB while reading SRAM

    // Zero-extend first so the low-bit select is legal even if M_W < AW.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [M_W+LRB_AW-1:0] m_ext = {{LRB_AW{1'b0}}, m_cnt};
    wire [M_W+ACT_AW-1:0] a_ext = {{ACT_AW{1'b0}}, m_cnt};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [LRB_AW-1:0]     idx   = m_ext[LRB_AW-1:0];

    assign act_rd_en   = iss_vld & ~use_lrb;
    assign act_rd_addr = a_ext[ACT_AW-1:0];
    assign lrb_rd_en   = iss_vld &  use_lrb;
    assign lrb_rd_addr = idx;

    // Side-band signals delayed one cycle to line up with registered
    // memory read data.
    reg               feed_vld_r, feed_sel_r, fill_r;
    reg [LRB_AW-1:0]  idx_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            feed_vld_r <= 1'b0;
            feed_sel_r <= 1'b0;
            fill_r     <= 1'b0;
            idx_r      <= {LRB_AW{1'b0}};
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
    // Result -> output SRAM
    //-------------------------------------------------------------------
    assign out_wr_en   = res_vld;
    assign out_wr_addr = out_wptr;

    //-------------------------------------------------------------------
    // Main FSM + counters
    //-------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            cfg_err  <= 1'b0;
            tile_cnt <= {T_W{1'b0}};
            t_lat    <= {T_W{1'b0}};
            m_lat    <= {M_W{1'b0}};
            m_cnt    <= {M_W{1'b0}};
            ld_cnt   <= {LD_W{1'b0}};
            wgt_base <= {WGT_AW{1'b0}};
            res_cnt  <= {M_W{1'b0}};
            out_wptr <= {OUT_AW{1'b0}};
        end else begin
            // Result bookkeeping (any state; DRAIN/LDW clears below win).
            if (res_vld) begin
                res_cnt  <= res_cnt + 1'b1;
                out_wptr <= out_wptr + 1'b1;
            end

            case (state)
                //---------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        done <= 1'b0;
                        if (cfg_ok) begin
                            cfg_err  <= 1'b0;
                            m_lat    <= cfg_m;
                            t_lat    <= cfg_tiles;
                            tile_cnt <= {T_W{1'b0}};
                            wgt_base <= {WGT_AW{1'b0}};
                            out_wptr <= {OUT_AW{1'b0}};
                            res_cnt  <= {M_W{1'b0}};
                            ld_cnt   <= {LD_W{1'b0}};
                            state    <= S_LDW;
                        end else begin
                            cfg_err  <= 1'b1;     // reject, stay idle
                        end
                    end
                end
                //---------------------------------------------------
                S_LDW: begin
                    if (ld_cnt == LD_LAST) begin
                        ld_cnt <= {LD_W{1'b0}};
                        m_cnt  <= {M_W{1'b0}};
                        state  <= S_COMP;
                    end else begin
                        ld_cnt <= ld_cnt + 1'b1;
                    end
                end
                //---------------------------------------------------
                S_COMP: begin
                    if (m_cnt == m_lat - 1'b1) begin
                        m_cnt <= {M_W{1'b0}};
                        state <= S_DRAIN;
                    end else begin
                        m_cnt <= m_cnt + 1'b1;
                    end
                end
                //---------------------------------------------------
                S_DRAIN: begin
                    if (res_cnt == m_lat) begin
                        if (tile_cnt == t_lat - 1'b1) begin
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            tile_cnt <= tile_cnt + 1'b1;
                            wgt_base <= wgt_base + ROWS;
                            res_cnt  <= {M_W{1'b0}};
                            ld_cnt   <= {LD_W{1'b0}};
                            state    <= S_LDW;
                        end
                    end
                end
                //---------------------------------------------------
                default: state <= S_IDLE;   // recover from illegal encoding
            endcase
        end
    end

endmodule
