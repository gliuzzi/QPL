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
	export run_codes
	
	#####################################################
	# Set the SEED of the random number generator
	#####################################################
	Random.seed!(28947)

	function warmup(nloc,MAXNODES,TOLGAP,time_limit)
		probfile = "qp20_10_1_1.mat"
		println(probfile)
		DATA = read_problem("../../mat/randqp/"*probfile)
		Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
		push!(DATA, "Qpos"     => Qpos)
		push!(DATA, "Dn"       => Dn)
		push!(DATA, "Un"       => Un)
		push!(DATA, "num_nneg" => num_nneg)

		#run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, false)

		#run_form7(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit)

		run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, false)

	end
	

	#res2  = open("results2.txt","a")
	#res7  = open("results7.txt","a")
	#res10 = open("results10.txt","a")
	#@printf(res2, "PROBLEM         |                 |     GAP |             GUB |  NODES |    TIME\n")
	#@printf(res2, "----------------+-----------------+---------+-----------------+--------+---------\n")
	#@printf(res7, "PROBLEM         |                 |     GAP |             GUB |  NODES |    TIME\n")
	#@printf(res7, "----------------+-----------------+---------+-----------------+--------+---------\n")
	#@printf(res10,"PROBLEM         |                 |     GAP |             GUB |  NODES |    TIME\n")
	#@printf(res10,"----------------+-----------------+---------+-----------------+--------+---------\n")
	#close(res2)
	#close(res7)
	#close(res10)


	#qp20_10_1_3.mat | FORM(10)        |      20 | 0.00000 |   -1.062169e+01 |     25 |   17.92
	#qp20_10_1_4.mat | FORM(10)        |      15 | 0.00000 |   -1.831369e+01 |     63 |   41.72
	#qp20_10_2_1.mat | FORM(10)        |      25 | 0.00000 |   -3.244199e+00 |     19 |   18.68

	#qp20_10_3_2.mat | FORM(10)        |      40 | 0.00000 |   -1.505081e+01 |      5 |    6.74
	#qp20_10_3_3.mat | FORM(10)        |      35 | 0.00000 |   -5.640822e+00 |     23 |   28.28
	#qp20_10_3_4.mat | FORM(10)        |      25 | 0.00000 |   -1.266500e+01 |     17 |   19.18

	#qp20_10_4_3.mat | FORM(10)        |      45 | 0.00000 |   -3.286259e+00 |      7 |    8.63
	#qp20_10_4_4.mat | FORM(10)        |      55 | 0.00000 |    1.921876e+00 |      7 |    7.40

	#qp30_15_2_3.mat | FORM(10)        |      47 | 0.00000 |   -2.069309e+00 |      3 |   23.42
	#qp30_15_2_4.mat | FORM(10)        |      77 | 0.00000 |    1.286245e+00 |      1 |    6.17
	#qp30_15_3_2.mat | FORM(10)        |      60 | 0.00000 |    1.447281e+01 |      1 |    4.19

	#qp30_15_3_4.mat | FORM(10)        |      43 | -0.00000 |   -2.934463e+00 |      1 |    4.88
	#qp30_15_4_3.mat | FORM(10)        |      73 | 0.00000 |    9.979285e+00 |      1 |    2.58
	#qp30_15_4_4.mat | FORM(10)        |      57 | 0.00000 |    6.601011e+00 |      1 |    2.62
	#qp40_20_1_2.mat | FORM(10)        |      60 | 0.00003 |    2.886951e-02 |      3 |   40.16
	#qp40_20_1_3.mat | FORM(10)        |      38 | 0.00000 |   -2.729287e+00 |      9 |   74.92
	#qp40_20_1_4.mat | FORM(10)        |      83 | 0.00000 |    1.205384e+01 |      1 |    5.06
	#qp40_20_2_2.mat | FORM(10)        |      85 | 0.00000 |    1.582996e+01 |      1 |   12.64

	#qp40_20_3_2.mat | FORM(10)        |      78 | 0.00000 |    8.941939e+00 |      1 |    5.78
	#qp40_20_3_3.mat | FORM(10)        |      53 | 0.00000 |    1.320789e+01 |      1 |    6.01
	#qp40_20_3_4.mat | FORM(10)        |      50 | 0.00068 |   -1.571793e+00 |     29 |  153.73
	#qp40_20_4_1.mat | FORM(10)        |      83 | 0.00000 |    4.559835e+01 |      1 |   12.25
	#qp50_25_1_2.mat | FORM(10)        |      66 | 0.00000 |    1.382087e+01 |      1 |    6.43
	#qp50_25_1_4.mat | FORM(10)        |      68 | 0.00000 |    1.384398e+01 |      1 |   11.43

	#qp50_25_3_2.mat | FORM(10)        |      60 | 0.00000 |    3.598633e+01 |      1 |    4.67
	#qp50_25_3_4.mat | FORM(10)        |      86 | 0.00000 |    1.529618e+01 |      1 |    6.99
	#qp50_25_4_1.mat | FORM(10)        |      80 | 0.00000 |    4.176818e+01 |      1 |   10.60
	#qp50_25_4_2.mat | FORM(10)        |      72 | 0.00000 |    2.881231e+01 |      1 |   15.50
	#qp50_25_4_4.mat | FORM(10)        |      60 | 0.00001 |    8.916406e+00 |      3 |   65.83

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
			@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(10)",num_nneg*100/nvar,GAP10,GUB10,totnodes10,execution_time10)
			close(res)
		end
		GC.gc()
	end
	
	function run_form7(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit)
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
		
		res  = open("results.txt","a")
		@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(7)",num_nneg*100/nvar,GAP2,GUB2,totnodes2,execution_time2)
		close(res)
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
			@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2)",num_nneg*100/nvar,GAP2,GUB2,totnodes2,execution_time2)
			close(res)
		end
	end

	function run_instance(n,n2,i1,i2,nloc, MAXNODES, TOLGAP, time_limit)

		probfile = "qp$(n)_$(n2)_$(i1)_$(i2).mat"
		println(probfile)
		DATA = read_problem("../../mat/randqp/"*probfile)
		Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
		push!(DATA, "Qpos"     => Qpos)
		push!(DATA, "Dn"       => Dn)
		push!(DATA, "Un"       => Un)
		push!(DATA, "num_nneg" => num_nneg)

		nvar, nvar2 = size(DATA["Q"])

		#if num_nneg*100/nvar > 60
		if false
			run_form10(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		else
			run_form2(probfile, DATA, nloc, MAXNODES, TOLGAP, time_limit, true)
		end
	end

	function run_codes(nloc,MAXNODES,TOLGAP,time_limit,triplets)


		for (n,i1,i2) in triplets
					n2 = Int(n/2)
					#####################################################
					# Set the SEED of the random number generator
					#####################################################
					Random.seed!(28947)
					
					run_instance(n,n2,i1,i2,nloc, MAXNODES, TOLGAP, time_limit)

					#println("Going to sleep...")
					#sleep(3)
					#println("...done")
					
					#if true
					#	res  = open("results.txt","a")
					#	@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(10-2)",num_nneg*100/nvar,GAP10,GUB10,totnodes10,execution_time10)
					#	close(res)
					#else
					#	res  = open("results.txt","a")
					#	@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2-10)",num_nneg*100/nvar,GAP2,GUB2,totnodes2,execution_time2)
					#	close(res)
					#end

					if true
					
					elseif true
					else
						(rootnodegap_time, GAP, prob2, LBs2, solved, error) = compute_rootnode_gap_7(DATA, nloc, MAXNODES, TOLGAP, time_limit)
						GUB = prob2.GUB
						totnodes = prob2.totnodes
						execution_time = 0.0
						if (!solved) && (!error)
							(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_7(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
						end
						execution_time += rootnodegap_time
						res  = open("results.txt","a")
						@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(7)",num_nneg*100/nvar,GAP,GUB,totnodes,execution_time)
						close(res)
					end

		end

		#println("Finito...... 2")
		#readline()
	end

	function run_codes2(nloc,MAXNODES,TOLGAP,time_limit,triplets,dims)
		res  = open("results.txt","a")
		@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		@printf(res,  "           select between FORM(2) and FORM(10) with gap at root node             \n")
		@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		@printf(res,  "                 con strong Bound tightening su tutti i nodi\n")
		@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		@printf(res,  "PROBLEM         |                 | rGAP(2) |rGAP(10) |     GAP |             GUB |  NODES |    TIME\n")
		@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		close(res)


		#for (n,i1,i2) in triplets
		for n in dims
			n2 = Int(n/2)
			for i1 in 1:4
				for i2 in 1:4
					#####################################################
					# Set the SEED of the random number generator
					#####################################################
					Random.seed!(28947)
					probfile = "qp$(n)_$(n2)_$(i1)_$(i2).mat"
					println(probfile)
					DATA = read_problem("../../mat/randqp/"*probfile)
					Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
					push!(DATA, "Qpos"     => Qpos)
					push!(DATA, "Dn"       => Dn)
					push!(DATA, "Un"       => Un)
					push!(DATA, "num_nneg" => num_nneg)

					nvar, nvar2 = size(DATA["Q"])

					#####################################################
					# Set the SEED of the random number generator
					#####################################################
					Random.seed!(28947)
					(rootnodegap_time2,  GAP2,  prob2,  LBs2,  solved2,  error2)  = compute_rootnode_gap_2(DATA, nloc, MAXNODES, TOLGAP, time_limit)
					if solved2
						GUB = prob2.GUB
						totnodes = prob2.totnodes
						execution_time = rootnodegap_time2
						res  = open("results.txt","a")
						@printf(res, "%-15s | %-15s | %7.5f |    --   | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2)",GAP2,GAP2,GUB,totnodes,execution_time)
						close(res)
					else
						#####################################################
						# Set the SEED of the random number generator
						#####################################################
						Random.seed!(28947)
						(rootnodegap_time10, GAP10, prob10, LBs10, solved10, error10) = compute_rootnode_gap_10(DATA, nloc, MAXNODES, TOLGAP, time_limit)
						if solved10
							GUB = prob10.GUB
							totnodes = prob10.totnodes
							execution_time = rootnodegap_time10 + rootnodegap_time2
							res  = open("results.txt","a")
							@printf(res, "%-15s | %-15s | %7.5f | %7.5f | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(10)",GAP2,GAP10,GAP10,GUB,totnodes,execution_time)
							close(res)
						else
							rootnodegap_time = rootnodegap_time2 + rootnodegap_time10

							if GAP10 < GAP2
								(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_10(DATA,prob10,LBs10,MAXNODES,TOLGAP,time_limit-rootnodegap_time10)
								execution_time += rootnodegap_time
								res  = open("results.txt","a")
								@printf(res, "%-15s | %-15s | %7.5f | %7.5f | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(10)",GAP2,GAP10,GAP,GUB,totnodes,execution_time)
								close(res)
							else
								(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_2(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit-rootnodegap_time2)
								execution_time += rootnodegap_time
								res  = open("results.txt","a")
								@printf(res, "%-15s | %-15s | %7.5f | %7.5f | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2)",GAP2,GAP10,GAP,GUB,totnodes,execution_time)
								close(res)
							end
						end
					end
				end
			end
		end

		println("Finito......")
		readline()

	end


	function run_codes3(nloc,MAXNODES,TOLGAP,time_limit,triplets,dims)
		res  = open("results.txt","a")
		@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		@printf(res,  "PROBLEM         |                 | rGAP(2) |rGAP(10) |     GAP |             GUB |  NODES |    TIME\n")
		@printf(res,  "----------------+-----------------+---------+-----------------+--------+---------\n")
		@printf(res,  "                 con strong Bound tightening su tutti i nodi\n")
		@printf(res,  "----------------+-----------------+---------+-----------------+--------+---------\n")
		#@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		#@printf(res,  "           select between FORM(2) and FORM(10) with gap at root node             \n")
		#@printf(res,  "----------------+-----------------+---------+---------+---------+-----------------+--------+---------\n")
		close(res)

		for n in dims
			n2 = Int(n/2)
			for i1 in 1:4
				for i2 in 1:4
					#####################################################
					# Set the SEED of the random number generator
					#####################################################
					Random.seed!(28947)
					probfile = "qp$(n)_$(n2)_$(i1)_$(i2).mat"
					println(probfile)
					DATA = read_problem("../../mat/randqp/"*probfile)
					Qpos, Dn, Un, num_nneg = decompose_Q(DATA["Q"])
					push!(DATA, "Qpos"     => Qpos)
					push!(DATA, "Dn"       => Dn)
					push!(DATA, "Un"       => Un)
					push!(DATA, "num_nneg" => num_nneg)

					nvar, nvar2 = size(DATA["Q"])

					#if num_nneg*100/nvar > 60
					if true
						(rootnodegap_time, GAP, prob10, LBs10, solved, error) = compute_rootnode_gap_10(DATA, nloc, MAXNODES, TOLGAP, time_limit)
						GUB = prob10.GUB
						totnodes = prob10.totnodes
						execution_time = 0.0
						if (!solved) && (!error)
							(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_10(DATA,prob10,LBs10,MAXNODES,TOLGAP,time_limit)
						end
						execution_time += rootnodegap_time
						res  = open("results.txt","a")
						@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(10)",num_nneg*100/nvar,GAP,GUB,totnodes,execution_time)
						close(res)
					else
						(rootnodegap_time, GAP, prob2, LBs2, solved, error) = compute_rootnode_gap_2(DATA, nloc, MAXNODES, TOLGAP, time_limit)
						GUB = prob2.GUB
						totnodes = prob2.totnodes
						execution_time = 0.0
						if (!solved) && (!error)
							(GUB,GAP,totnodes,numlp,nummilp,execution_time,time_gap) = solve_2(DATA,prob2,LBs2,MAXNODES,TOLGAP,time_limit)
						end
						execution_time += rootnodegap_time
						res  = open("results.txt","a")
						@printf(res, "%-15s | %-15s | %7d | %7.5f | %15.6e | %6d | %7.2f\n",probfile,"FORM(2)",num_nneg*100/nvar,GAP,GUB,totnodes,execution_time)
						close(res)

					end

				end
			end
		end

		println("Fine")
		nothing
	end
end