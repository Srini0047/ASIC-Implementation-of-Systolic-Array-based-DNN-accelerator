# ModelSim: same test through the AXI4-Lite bridge.   vsim -c -do cadence/run_modelsim_axi.do
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work tb/tb_dnn_player.v -f cadence/rtl_files.f
vsim -c -GAXIL=1 work.tb_dnn_player +axi_stall +prog=build/prog.hex +log=build/axil_readback.txt
run -all
quit -f
