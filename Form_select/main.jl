#####################################################
using Distributed
@everywhere push!(LOAD_PATH, pwd())
@everywhere using partools_form2
@everywhere using partools_form10

using mod_run

nloc       = 10
MAXNODES   = Inf
TOLGAP     = 1.e-5
time_limit = 1800
probfile   = "qp20_10_1_2.mat"

run_instance(probfile, nloc, MAXNODES, TOLGAP, time_limit)

nothing
