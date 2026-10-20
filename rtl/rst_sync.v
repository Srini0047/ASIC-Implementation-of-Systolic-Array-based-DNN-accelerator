//=============================================================================
// Module      : rst_sync
// Description : Reset synchronizer: asynchronous assertion, synchronous
//               de-assertion. This is the ONLY place in the design where an
//               asynchronous reset is used (documented exception); everything
//               downstream uses the synchronized, synchronous-style reset.
//
//   Clock-domain note: the accelerator is a single-clock design, so the only
//   asynchronous input that needs handling is the external reset. All host
//   inputs (start, cfg_*, host_*) must be launched from the same clk domain
//   (or be synchronized by the SoC wrapper).
//
//   The clock must be running while reset is asserted for the downstream
//   synchronous resets to take effect.
//   Set the STAGES >= 2. Mark the first stage with set_false_path / ASYNC_REG
//   equivalent attributes in your SDC/flow if your methodology requires.
//=============================================================================
module rst_sync #(
    parameter STAGES = 2
)(
    input  wire clk,
    input  wire arst_n,        // asynchronous, active low
    output wire srst_n         // synchronously released, active low
);

    reg [STAGES-1:0] sr;

    always @(posedge clk or negedge arst_n) begin
        if (!arst_n) sr <= {STAGES{1'b0}};
        else         sr <= (sr << 1) | {{(STAGES-1){1'b0}}, 1'b1};
    end

    assign srst_n = sr[STAGES-1];

endmodule
