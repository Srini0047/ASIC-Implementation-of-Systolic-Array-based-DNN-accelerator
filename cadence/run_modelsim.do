# ModelSim: normal batch run.   From the PROJECT ROOT folder type:    vsim -c -do cadence/run_modelsim.do
if {[file exists work]} { vdel -lib work -all }
vlib work
# testbench first (it carries the `timescale), then the design files
vlog -work work tb/tb_dnn_player.v -f cadence/rtl_files.f
vsim -c work.tb_dnn_player +prog=build/prog.hex +log=build/rtl_readback.txt
run -all
quit -f
