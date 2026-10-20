//=============================================================================
// Module      : perf_counters
// Description : Optional performance/telemetry counters used to evaluate the
//               report's "throughput, latency and data-reuse" metrics and to
//               quantify the LRB benefit (SRAM reads avoided).
//
//   All counters clear when `clear` pulses (start accepted while idle) and
//   wrap at 2^32. Instantiate with PERF_EN=0 at the top to strip them for
//   tape-out (saves 4 x 32 flops + incrementers).
//     cycles      : cycles while busy
//     act_sram_rd : activation-SRAM reads   (energy-expensive accesses)
//     lrb_rd      : LRB reads               (energy-cheap accesses)
//     wgt_sram_rd : weight-SRAM reads
//=============================================================================
module perf_counters (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        busy,
    input  wire        act_rd,
    input  wire        lrb_rd,
    input  wire        wgt_rd,
    output reg  [31:0] cycles,
    output reg  [31:0] act_sram_rd,
    output reg  [31:0] lrb_rd_cnt,
    output reg  [31:0] wgt_sram_rd
);

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            cycles      <= 32'd0;
            act_sram_rd <= 32'd0;
            lrb_rd_cnt  <= 32'd0;
            wgt_sram_rd <= 32'd0;
        end else begin
            if (busy)   cycles      <= cycles + 32'd1;
            if (act_rd) act_sram_rd <= act_sram_rd + 32'd1;
            if (lrb_rd) lrb_rd_cnt  <= lrb_rd_cnt + 32'd1;
            if (wgt_rd) wgt_sram_rd <= wgt_sram_rd + 32'd1;
        end
    end

endmodule
