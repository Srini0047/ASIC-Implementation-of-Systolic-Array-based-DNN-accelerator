# ModelSim: the tiny hand-checkable example (4 numbers in, 4 numbers out).
#   1) python sw\example_tiny_net.py          (makes build\tiny_prog.hex; it is also already shipped)
#   2) vsim -c -do cadence/run_modelsim_tiny.do
#   3) python sw\show_tiny.py                 (prints what the chip answered)
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work tb/tb_dnn_player.v -f cadence/rtl_files.f
vsim -c work.tb_dnn_player +prog=build/tiny_prog.hex +log=build/custom_readback.txt
run -all
quit -f
