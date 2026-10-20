# ModelSim GUI for the single-PE testbench:  vsim -do cadence/run_pe_modelsim_gui.do
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work tb/tb_pe.v rtl/pe.v
vsim -voptargs=+acc work.tb_pe
add wave -divider {inputs driven by the testbench}
add wave /tb_pe/clk
add wave /tb_pe/rst_n
add wave /tb_pe/en
add wave /tb_pe/weight_load
add wave -radix decimal /tb_pe/weight_in
add wave -radix decimal /tb_pe/act_in
add wave -radix decimal /tb_pe/psum_in
add wave -divider {outputs of the PE}
add wave -radix decimal /tb_pe/weight_out
add wave -radix decimal /tb_pe/act_out
add wave -radix decimal /tb_pe/psum_out
onfinish stop
run -all
wave zoom full
