#####################################################
using Distributed
@everywhere push!(LOAD_PATH, pwd())

using Printf
using Random
using utility_form2
using utility_form7
using utility_form10
using utility

#####################################################
# Set the SEED of the random number generator
#####################################################
Random.seed!(28947)

#####################################################
# The user must specify
#
# probfileS : name of file (with full path) with S matrix in CSV format
# probfileA : name of file (with full path) with A matrix in CSV format
# nloc 	    : number of initial local minimizations to set the initial GUB
# MAXNODES  : maximum number of B&B nodes (a number >= 0, possibly Inf)
#             N.B. setting MAXNODES to Inf could result in an extremely
#             long time to complete the run, use with caution
#
#####################################################

nloc       = 10
MAXNODES   = Inf
TOLGAP     = 1.e-5
time_limit = 1800

	probfile = "qp20_10_1_1.mat"
	println(probfile)
	DATA = read_problem("../../mat/randqp/"*probfile)
	Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
	push!(DATA, "Qpos"     => Qpos)
	push!(DATA, "Dn"       => Dn)
	push!(DATA, "Un"       => Un)
	push!(DATA, "num_nneg" => num_nneg)

	(rootnodegap_time, gap, prob2, LBs2, solved, error) = compute_rootnode_gap_2(DATA, nloc, MAXNODES, TOLGAP, time_limit)
	if (!solved) && (!error)
		(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_2(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
	end

dims = [20, 30, 40, 50]
#dims = [30]

res2  = open("results2.txt","a")
@printf(res2, "----------------+-----------------+---------+-----------------+--------+---------\n")
@printf(res2, "PROBLEM         |                 |     GAP |             GUB |  NODES |    TIME\n")
@printf(res2, "----------------+-----------------+---------+-----------------+--------+---------\n")
@printf(res2, "         0 Bound tightening soltanto a tutti i nodi\n")
@printf(res2, "----------------+-----------------+---------+-----------------+--------+---------\n")
close(res2)

for n in dims
	n2 = Int(n/2)
	for i1 in 1:4
		for i2 in 1:4
			probfile = "qp$(n)_$(n2)_$(i1)_$(i2).mat"
			println(probfile)
			DATA = read_problem("../../mat/randqp/"*probfile)

			(rootnodegap_time, GAP, prob2, LBs2, solved, error) = compute_rootnode_gap_2(DATA, nloc, MAXNODES, TOLGAP, time_limit)
			GUB = prob2.GUB
			totnodes = prob2.totnodes
			execution_time = 0.0
			if (!solved) && (!error)
				(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_2(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
			end
			execution_time += rootnodegap_time
			res  = open("results2.txt","a")
			@printf(res, "%-15s | %-15s | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2)",GAP,GUB,totnodes,execution_time)
			close(res)

		end
	end
end


println("Fine")
nothing
