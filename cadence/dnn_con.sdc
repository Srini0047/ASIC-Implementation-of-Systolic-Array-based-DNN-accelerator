# Constraints for dnn_accel_top, in the style of your lab's con.sdc.
# The ports are the real ports of dnn_accel_top: clk, rst_n_i, reg_wr_en, reg_rd_en, reg_addr[15:0], reg_wdata[31:0],
# reg_rdata[31:0], irq.
# Start RELAXED (10 ns = 100 MHz). When timing is met (slack >= 0 in dnn_timing.rpt), lower the period and re-run.
create_clock -name clk -period 10 [get_ports clk]
set_clock_transition -rise 0.1 [get_clocks clk]
set_clock_transition -fall 0.1 [get_clocks clk]
set_clock_uncertainty 0.1 [get_clocks clk]

# inputs: the register bus (the outside world uses 40 % of the period); outputs likewise
set_input_delay  -max 4 [get_ports {reg_wr_en reg_rd_en reg_addr[*] reg_wdata[*]}] -clock [get_clocks clk]
set_output_delay -max 4 [get_ports {reg_rdata[*] irq}] -clock [get_clocks clk]

# asynchronous reset pin: it only feeds the 2-flop reset synchroniser, so it is not timed
set_false_path -from [get_ports rst_n_i]

group_path -name   I2R   -from   [all_inputs]      -to   [all_registers]
group_path -name   R2O   -from   [all_registers]   -to   [all_outputs]
group_path -name   R2R   -from   [all_registers]   -to   [all_registers]
group_path -name   I2O   -from   [all_inputs]      -to   [all_outputs]
