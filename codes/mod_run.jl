__precompile__()

	module mod_run
	#####################################################
	using Distributed
	@everywhere push!(LOAD_PATH, pwd())

	using Printf
	using Random
	using utility_form2
	using utility_form7
	using utility_form10
	using utility

	export warmup
	export run_instance

	#####################################################
	# Set the SEED of the random number generator
	#####################################################
	Random.seed!(28947)

	function warmup(nloc,MAXNODES,TOLGAP,time_limit)
		probfile = "qp20_10_1_1.mat"
		println(probfile)
		DATA = read_problem("../randqp/"*probfile)
		Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
		push!(DATA, "Qpos"     => Qpos)
		push!(DATA, "Dn"       => Dn)
		push!(DATA, "Un"       => Un)
		push!(DATA, "num_nneg" => num_nneg)

		#run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, false)

		#run_form7(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit)

		run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, false)

	end

	function run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, stampa)
		nvar, nvar2 = size(DATA["Q"])
		num_nneg = DATA["num_nneg"]

		prob10 = 0
		LBs10  = 0
		#####################################################
		# Set the SEED of the random number generator
		#####################################################
		Random.seed!(28947)
		(rootnodegap_time10, GAP10, prob10, LBs10, solved, error) = compute_rootnode_gap_10(DATA, nloc, MAXNODES, TOLGAP, time_limit)
		GUB10 = prob10.GUB
		totnodes10 = prob10.totnodes
		execution_time10 = 0.0
		if (!solved) && (!error)
			(GUB10,GAP10,totnodes10,numlp10,nummilp,execution_time10,time_gap) = solve_10(DATA,prob10,LBs10,MAXNODES,TOLGAP,time_limit-rootnodegap_time10)
		end
		execution_time10 += rootnodegap_time10

		if stampa
			res  = open("results.txt","a")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res,  "PROBLEM         | FORMULATION     |    nneg |     GAP |             GUB |  NODES |    TIME\n")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"B&T(Conv)",num_nneg*100/nvar,GAP10,GUB10,totnodes10,execution_time10)
			close(res)
		end
		GC.gc()
	end

	function run_form7(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, stampa)
		nvar, nvar2 = size(DATA["Q"])
		num_nneg = DATA["num_nneg"]

		#####################################################
		# Set the SEED of the random number generator
		#####################################################
		Random.seed!(28947)
		(rootnodegap_time2, GAP2, prob2, LBs2, solved, error) = compute_rootnode_gap_7(DATA, nloc, MAXNODES, TOLGAP, time_limit)
		GUB2 = prob2.GUB
		totnodes2 = prob2.totnodes
		execution_time2 = 0.0
		if (!solved) && (!error)
			(GUB2,GAP2,totnodes2,numlp2,nummilp,execution_time2,time_gap) = solve_7(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
		end
		execution_time2 += rootnodegap_time2

		if stampa
			res  = open("results.txt","a")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res,  "PROBLEM         | FORMULATION     |    nneg |     GAP |             GUB |  NODES |    TIME\n")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"B&T(XY)",num_nneg*100/nvar,GAP2,GUB2,totnodes2,execution_time2)
			close(res)
		end
	end

	function run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, stampa)
		nvar, nvar2 = size(DATA["Q"])
		num_nneg = DATA["num_nneg"]

		#####################################################
		# Set the SEED of the random number generator
		#####################################################
		Random.seed!(28947)
		(rootnodegap_time2, GAP2, prob2, LBs2, solved, error) = compute_rootnode_gap_2(DATA, nloc, MAXNODES, TOLGAP, time_limit)
		GUB2 = prob2.GUB
		totnodes2 = prob2.totnodes
		execution_time2 = 0.0
		if (!solved) && (!error)
			(GUB2,GAP2,totnodes2,numlp2,nummilp,execution_time2,time_gap) = solve_2(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
		end
		execution_time2 += rootnodegap_time2

		if stampa
			res  = open("results.txt","a")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res,  "PROBLEM         | FORMULATION     |    nneg |     GAP |             GUB |  NODES |    TIME\n")
			@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
			@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"B&T(Bil)",num_nneg*100/nvar,GAP2,GUB2,totnodes2,execution_time2)
			close(res)
		end
	end

	function run_instance(probfile, nloc, MAXNODES, TOLGAP, time_limit)
		println(probfile)
		DATA = read_problem("../randqp/"*probfile)
		Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
		push!(DATA, "Qpos"     => Qpos)
		push!(DATA, "Dn"       => Dn)
		push!(DATA, "Un"       => Un)
		push!(DATA, "num_nneg" => num_nneg)

		nvar, nvar2 = size(DATA["Q"])

		if num_nneg*100/nvar > 60
			run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		else
			run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		end
	end

	function run_instance(n,n2,i1,i2,nloc, MAXNODES, TOLGAP, time_limit)
		probfile = "qp$(n)_$(n2)_$(i1)_$(i2).mat"
		println(probfile)
		DATA = read_problem("../randqp/"*probfile)
		Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
		push!(DATA, "Qpos"     => Qpos)
		push!(DATA, "Dn"       => Dn)
		push!(DATA, "Un"       => Un)
		push!(DATA, "num_nneg" => num_nneg)

		nvar, nvar2 = size(DATA["Q"])

		if num_nneg*100/nvar > 60
			run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		else
			run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		end
	end
end
