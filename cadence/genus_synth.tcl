# Genus synthesis TEMPLATE for dnn_accel_top.  NOT run (no Cadence tools were available where this was written).
# Run from the PROJECT ROOT folder:      genus -f cadence/genus_synth.tcl -log genus.log
# Fill in the two lines marked <<<  with what your lab / tool administrator gives you.

set DESIGN     dnn_accel_top
set LIB_DIR    "<<< FOLDER that contains the standard-cell .lib files, e.g. /pdk/tsmc/lib/ >>>"
set LIB_FILES  "<<< the .lib file name(s), e.g. slow.lib >>>"

set_db init_lib_search_path $LIB_DIR
set_db init_hdl_search_path ./rtl
set_db library $LIB_FILES
# optional: let Genus insert clock gates where the design has enables (the datapath is gated by 'busy')
# set_db lp_insert_clock_gating true

read_hdl -v2001 {          ;# if Genus complains about $clog2, change -v2001 to -sv
  pe.v skew_delay.v systolic_array.v sa_core.v sram_sdp.v lrb.v perf_counters.v rst_sync.v
  dnn_ppu.v dnn_ctrl.v dnn_accel_core.v dnn_reg_if.v dnn_accel_top.v
}
elaborate $DESIGN
check_design -unresolved          ;# must list nothing unresolved
read_sdc cadence/constraints.sdc

set_db syn_generic_effort medium
syn_generic                        ;# RTL -> generic gates
set_db syn_map_effort medium
syn_map                            ;# generic gates -> your library cells
set_db syn_opt_effort medium
syn_opt                            ;# fix timing / recover area

file mkdir genus_out
report_timing > genus_out/timing.rpt      ;# worst path: is slack >= 0 ?
report_area   > genus_out/area.rpt        ;# how big
report_power  > genus_out/power.rpt       ;# rough power (no activity file => statistical guess)
report_gates  > genus_out/gates.rpt       ;# cell-type counts
write_hdl > genus_out/${DESIGN}_netlist.v ;# gate-level netlist (input to Innovus)
write_sdc > genus_out/${DESIGN}.sdc
puts "DONE - open genus_out/timing.rpt, area.rpt, power.rpt"
exit
