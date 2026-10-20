# ModelSim GUI with the interesting signals already on the waveform.   vsim -do cadence/run_modelsim_gui.do
# (first batch only: probe + weight load + 8 images + read-back, about 2400 clock cycles)
if {[file exists work]} { vdel -lib work -all }
vlib work
vlog -work work tb/tb_dnn_player.v -f cadence/rtl_files.f
vsim -voptargs=+acc work.tb_dnn_player +prog=build/wave_prog.hex +log=build/wave_readback.txt
add wave -divider {clock / register bus (what the "CPU" does)}
add wave /tb_dnn_player/clk
add wave /tb_dnn_player/reg_wr_en
add wave -radix hex /tb_dnn_player/reg_addr
add wave -radix hex /tb_dnn_player/reg_wdata
add wave -divider {controller: what the chip is doing}
add wave -radix unsigned /tb_dnn_player/g_reg/dut/u_core/u_ctrl/state
add wave -radix unsigned /tb_dnn_player/g_reg/dut/u_core/u_ctrl/layer_cnt
add wave -radix unsigned /tb_dnn_player/g_reg/dut/u_core/u_ctrl/kt_cnt
add wave -radix unsigned /tb_dnn_player/g_reg/dut/u_core/u_ctrl/nt_cnt
add wave -radix unsigned /tb_dnn_player/g_reg/dut/u_core/u_ctrl/m_cnt
add wave /tb_dnn_player/g_reg/dut/u_core/u_ctrl/busy
add wave /tb_dnn_player/g_reg/dut/u_core/u_ctrl/done
add wave /tb_dnn_player/g_reg/dut/u_core/u_ctrl/err
add wave -divider {systolic array and accumulate}
add wave /tb_dnn_player/g_reg/dut/u_core/u_core/res_vld
add wave -radix decimal /tb_dnn_player/g_reg/dut/u_core/u_core/res_vec
add wave /tb_dnn_player/g_reg/dut/u_core/acc_wr_en
add wave /tb_dnn_player/g_reg/dut/u_core/acc_rd_en
onfinish stop          ;# keep ModelSim open when the testbench ends (default would ask "Are you sure you want to finish?")
run -all
wave zoom full
