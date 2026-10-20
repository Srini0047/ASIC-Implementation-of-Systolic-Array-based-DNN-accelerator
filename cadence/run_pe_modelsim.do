# ModelSim, console only:  vsim -c -do cadence/run_pe_modelsim.do      (the .bat file does this for you)
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work tb/tb_pe.v rtl/pe.v
vsim -c work.tb_pe
run -all
quit -f
