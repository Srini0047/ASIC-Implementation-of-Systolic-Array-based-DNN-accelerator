# Timing constraints for dnn_accel_top  (TEMPLATE - not run through any Cadence tool; read docs/BEGINNER_GUIDE.html, section 9, Job 4)
# An SDC file tells the synthesis tool the speed you want and what the outside world looks like.

# 1) The clock: period 10 ns = 100 MHz. Start relaxed. When synthesis meets it easily, lower this number and re-run.
set clk_period 10.0
create_clock -name clk -period $clk_period [get_ports clk]
set_clock_uncertainty 0.2 [get_clocks clk]
set_clock_transition  0.1 [get_clocks clk]

# 2) The register bus is clocked by clk; assume the outside logic uses 40 % of the period.
set_input_delay  -clock clk [expr 0.4*$clk_period] [get_ports {reg_wr_en reg_rd_en reg_addr[*] reg_wdata[*]}]
set_output_delay -clock clk [expr 0.4*$clk_period] [get_ports {reg_rdata[*] irq}]

# 3) The asynchronous reset pin only feeds the 2-flop reset synchronizer; do not time it.
set_false_path -from [get_ports rst_n_i]

# 4) Pin electrical assumptions. Replace BUFX2 by a real buffer cell name from YOUR library, or delete this line.
# set_driving_cell -lib_cell BUFX2 [all_inputs]
set_load 0.05 [all_outputs]
