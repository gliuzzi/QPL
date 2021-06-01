#####################################################
using Distributed
@everywhere push!(LOAD_PATH, pwd())

using mod_run

nloc       = 10					# number of local searches to determine first GUB
MAXNODES   = Inf				# maximum number of nodes to be explored
TOLGAP     = 1.e-5				# tolerance for node fathoming and stop
time_limit = 1800				# time limit
probfile   = "qp20_10_1_2.mat"  # problem file in folder ../randqp

addprocs(10)
    @everywhere push!(LOAD_PATH, pwd())
    @everywhere using partools_form2
    @everywhere using partools_form10
    run_instance(probfile, nloc, MAXNODES, TOLGAP, time_limit)
rmprocs(workers(),waitfor=0)


nothing
