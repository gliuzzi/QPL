__precompile__()

module utility_form7
	using Distributed
	using BB_form7
	@everywhere using partools_form7
	using Gurobi
	using Ipopt
	using MAT
	using LinearAlgebra
	using Printf
	using Random

	#####################################################
	# Set the SEED of the random number generator
	#####################################################
	Random.seed!(28947)

	export solve_7
	export compute_rootnode_gap_7

	function compute_rootnode_gap_7(DATA,
					 nloc::Int,
					 MAXNODESin::Number,
					 TOLGAP_IN::Number,
					 time_limit)

		iprint = 0
		#####################################################
		# The main program STARTS HERE
		#####################################################
		# instantiate the Branch & Bound object and define:
		#	management of lp's at B&B nodes
		# 	policy for management of list of open problems;
		#	when the LB for an open problem is computed.
		#	rebuild can take values in {true, false}
		#	policy can take values in {:lifo, :fifo, :sort}
		#	whenlb can take values in {:before, :after}
		#####################################################
		# rebuild  : tells whether LP problems at the B&B tree nodes must be recomputed
		#	     from scratch every time an lb must be computed
		# policy   : defines how the queue is managed. Allowed values are {:lifo, :sort}
		#	     :sort means best-bound visit of B&B tree
		#  		     :lifo means depth-first visit of B&B tree
		# branch   : kind of branch strategy. Allowed values are {:binary, :nary}
		#####################################################

		rebuild  = false
		policy   = :sort
		######################################################
		# NOTE: at the moment branch MUST be :binary
		######################################################
		branch   = :binary

		if(MAXNODESin <= 0)
			println("WARNING: passed value for MAXNODES is <= 0, B&B will explore 0 nodes!")
			MAXNODES = 0
		else
			MAXNODES = MAXNODESin
		end
		if(TOLGAP_IN > 0)
			TOL_GAP = TOLGAP_IN
		end

		prob         = BB_form7.BB_7()
		LBs          = Array{Float64}(undef,0)

		prob.rebuild = rebuild
		prob.policy  = policy
		prob.branch  = branch

		if !(prob.policy in BB_form7.POLICY_VALUES)
			error("ERROR!: possible values for policy are ",transpose(BB_form7.POLICY_VALUES),"\n")
		end
		if !(prob.whenlb in BB_form7.WHENLB_VALUES)
			error("ERROR!: possible values for whenlb are ",transpose(BB_form7.WHENLB_VALUES),"\n")
		end
		if !(prob.branch in BB_form7.BRANCH_VALUES)
			error("ERROR!: possible values for branch are ",transpose(BB_form7.BRANCH_VALUES),"\n")
		end
		if(policy == :sort)
			prob.whenlb = :after
		else
			prob.whenlb = :before
		end

		################ INPUT DATA   #######################
		#####################################################
		# Reading problem data from specified path
		#####################################################
		# problem:
		#
		# min 0.5 x'Qx + c'x + T
		# s.t. Ax <= b, Aeq x = beq
		#      LB <= x <= UB
		#
		LB   = DATA["LB"]
		UB   = DATA["UB"]
		c    = DATA["c"]
		Q    = DATA["Q"]
		A    = DATA["A"]
		b    = DATA["b"]
		Aeq  = DATA["Aeq"]
		beq  = DATA["beq"]
		T    = DATA["T"]
		flag = DATA["flag"]

		if !(flag)
			println("Some upper or lower bound on the variables is set to infinity. Exiting \n\n")
			solved = false
			error  = true
			return 0.0, Inf, prob, LBs, solved, error
		end
		(n,~)   = size(Q)
		(m,~)   = size(A)
		(meq,~) = size(Aeq)
		p       = convert(Int,n*(n+1)/2)

		Qvec = zeros(p)
		k = 0
		for j = 1:n
			for i = 1:j
				k += 1
				if i == j
					Qvec[k] = 0.5*Q[i,j]
				else
					Qvec[k] = Q[i,j]
				end
			end
		end

		prob.n   = n
		prob.m   = m
		prob.Q   = Q
		prob.Qvec= Qvec
		prob.c   = c
		prob.T   = T
		prob.A   = A
		prob.Aeq = Aeq
		prob.b   = b
		prob.beq = beq


		prob.nvar  = convert(Int,n + p)
		prob.vmap  = Dict("x" => 1:n,
						  "g" => n+1:n+p)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_7, [n for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
						   [Qvec for i in 1:nprocs()],
						   [c for i in 1:nprocs()],
						   [T for i in 1:nprocs()],
						   [A for i in 1:nprocs()],
						   [Aeq for i in 1:nprocs()],
						   [b for i in 1:nprocs()],
						   [beq for i in 1:nprocs()],
						   [prob.vmap for i in 1:nprocs()])
		######## SET DATA FOR PARALLEL PROCESSING END #######

		fid_GUB = open("GUB_stat.txt","w")
		fid_tim = open("tim_stat.txt","w")
		fid_bab = open("bab_stat.txt","w")

		println(fid_GUB, "  node LP's rebuild? ",prob.rebuild)
		println(fid_GUB, "   BB queue policy = ",prob.policy)
		println(fid_GUB, "            whenlb = ",prob.whenlb)
		println(fid_GUB, "branching strategy = ",prob.branch,"\n")

		begin

			start_time = time()
			GAP_time = 0.0
			GAP_node = 0
			if (nloc >0)
				(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,nloc)
				#xstar = [0.25838044394163306, 0.0, 0.0, 0.19120247537184457, 0.0, 0.3433121750327294, 0.0, 0.0, 0.6488588197152324, 0.0, 0.0, 0.0, 0.0, 0.573212917186699, 0.0, 0.0, 0.06177391227639872, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02199876127776128, 0.07204154466034857, 0.1694256448634834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37803804931411833, 0.0, 0.7416802440298111, 0.0, 0.0, 0.9107640307578458, 0.0, 0.0, 0.0, 0.0, 0.23640045712434524, 0.0, 0.0, 0.0, 0.0]
				#bestval = transpose(xstar)*Q*xstar/2 + (transpose(c)*xstar)[1] + T[1]
				prob.GUB = bestval
				prob.xUB = copy(xstar)
			else
				prob.GUB = Inf
			end
			tol           = TOL_GAP
			upperbound1   = prob.GUB
			upperbound    = (1-tol)*prob.GUB
			upperboundmod = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)
			println("build initial model (w/o Qbounds and MC constrs)")
			lp = build_initial_model_7(Q,prob.Qvec,c,T,A,b,Aeq,beq,LB,UB,upperboundmod)
			println("adding Qbounds and MC constrs to the lp")
			add_MC_cons!(lp,n,LB,UB,Q)

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = find_feasible_point(lp,prob)
				show(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				upperbound = (1-tol)*prob.GUB
				upperboundmod = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(xstar)
				lp = build_model_7(Q,prob.Qvec,c,T,A,b,Aeq,beq,LB,UB,upperboundmod)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end

			e = elem_bb_7(n)
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,lowx,upx,iprint)
			#println("    LB = ",lb,"     GUB = ",prob.GUB)
			#if iprint >= 1
			#	println("                  GUB = ",prob.GUB)
			#	println("                   UB = ",e.UB)
			#end
			(lb, ub, solvetime) = compute_lowerbound!(e,prob,tol,iprint,true)
			println("    LB = ",lb,"     GUB = ",prob.GUB)
			if iprint >= 1
				println("                  GUB = ",prob.GUB)
				println("                   UB = ",e.UB)
				println("lb at root node = ", lb,      "   e.UB = ",e.UB)
				println("            e.x = ",e.x)
			end

			rootgap_time = time()-start_time
			prob.GAP = (prob.GUB - lb)/max(1.0,abs(prob.GUB))

			@printf(fid_bab," lb at root node is %13.6e\n",lb)
			flush(fid_bab)
			@printf(fid_tim,"solution of LP took %20f seconds for node at level %10d \n",solvetime,e.level)
			flush(fid_tim)

			@printf(" lb at root node is %13.6e\n",lb)
			@printf(" ub at root node is %13.6e\n",prob.GUB)

			prob.totnodes += 1

			if (e.LB == Inf)
				execution_time = time()-start_time

				#write_final_stat(fid_STA,fid_GUB,prob,[],execution_time)

				if iprint > 2
					println(" xUB = ",prob.xUB)
				end

				close(fid_GUB)
				close(fid_tim)
				close(fid_bab)

				solved = true
				error  = false
				return (rootgap_time,0.0,prob,LBs,solved,error)

			end

			prob.GLB = e.LB
			if e.UB < prob.GUB
				prob.GUB = e.UB
				prob.xUB = copy(e.x)
				@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
				flush(fid_GUB)
			end

			push!(prob.open_probs,e)
			push!(LBs,e.LB)
			prob.nnodes += 1

			if(prob.policy == :sort)
				(BB_form7.sort!)(prob,(>))
			end
			prob.GLB = minimum(LBs)

			@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
			flush(fid_GUB)

			@printf(" open    tot          GUB          GAP             father info           ch.#           child info\n")
			@printf("                                         (     LB              UB    )        (     LB              UB    )\n")
			@printf(fid_bab," open    tot          GUB          GAP             father info           ch.#           child info\n")
			@printf(fid_bab,"                                         (     LB              UB    )        (     LB              UB    )\n")

			@printf(" %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d \n",
				 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,e.LB,e.UB,0)
			@printf(fid_bab," %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d \n",
				 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,e.LB,e.UB,0)
			flush(fid_bab)
			close(fid_GUB)
			close(fid_tim)
			close(fid_bab)

			solved = false
			error  = false
			return rootgap_time, prob.GAP, prob, LBs, solved, error
		end
	end

	function solve_7(DATA,prob,LBs,
					 MAXNODESin::Number,
					 TOLGAP_IN::Number,
					 time_limit)

		iprint = 0

		if(MAXNODESin <= 0)
			println("WARNING: passed value for MAXNODES is <= 0, B&B will explore 0 nodes!")
			MAXNODES = 0
		else
			MAXNODES = MAXNODESin
		end
		if(TOLGAP_IN > 0)
			TOL_GAP = TOLGAP_IN
		end

		LB   = DATA["LB"]
		UB   = DATA["UB"]
		c    = DATA["c"]
		Q    = DATA["Q"]
		A    = DATA["A"]
		b    = DATA["b"]
		Aeq  = DATA["Aeq"]
		beq  = DATA["beq"]
		T    = DATA["T"]
		flag = DATA["flag"]

		if !(flag)
			println("Some upper or lower bound on the variables is set to infinity. Exiting \n\n")
			return
		end
		(n,~)   = size(Q)
		(m,~)   = size(A)
		(meq,~) = size(Aeq)
		p       = convert(Int,n*(n+1)/2)


		fid_GUB = open("GUB_stat.txt","w")
		fid_tim = open("tim_stat.txt","w")
		fid_bab = open("bab_stat.txt","w")

		println(fid_GUB, "  node LP's rebuild? ",prob.rebuild)
		println(fid_GUB, "   BB queue policy = ",prob.policy)
		println(fid_GUB, "            whenlb = ",prob.whenlb)
		println(fid_GUB, "branching strategy = ",prob.branch,"\n")

		begin

			start_time = time()
			GAP_time = 0.0
			GAP_node = 0
			tol           = TOL_GAP
			upperbound1   = prob.GUB
			upperbound    = (1-tol)*prob.GUB
			upperboundmod = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)

			while (prob.nnodes > 0) && (prob.totnodes < MAXNODES) && (time()-start_time <= time_limit)
				sel_prob = BB_form7.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/max(1.0,abs(prob.GUB)) >=  -TOL_GAP)
					println("Fathom sel_prob since ", sel_prob.LB, " >= ", prob.GUB*(1 -TOL_GAP))
					continue
				end
				begin

					local child = subdivide(prob,sel_prob,iprint)
					nchild = length(child)
					index = sel_prob.index
					if iprint >= 1
						@printf("Selected  node = %8d, Lower bound = %13.6e, GUB = %13.6e\n",i,sel_prob.LB,prob.GUB)
						@printf("Percentage GAP = %8.6f, Elapsed tim = %13.6e, numnodes = %6d, open nodes = %6d\n",prob.GAP,time()-start_time,prob.totnodes,prob.nnodes)
						@printf("Branching  var = %8d,      x value= %13.6e, xmn = %13.6e, xmx = %13.6e\n",
								sel_prob.index,sel_prob.x[index],sel_prob.xmin[index],sel_prob.xmax[index])
					end
					if iprint > 2
						print("hit RETURN to continue ...")
						readline()
					end

					if nchild > 0
						solvetime = [0.0;0.0]
						lb = [0.0; 0.0]
						ub = [0.0; 0.0]
						num_lb = [0; 0]
						#Threads.@threads for i = 1:nchild
						#println(Threads.threadid())
						for i = 1:nchild
							#a = child[i]
							child[i].level = (sel_prob.level)+1

							if (child[i].LB < Inf)
								(lb[i], ub[i], solvetime[i]) = compute_lowerbound!(child[i],prob,tol,iprint,false)
								@printf(fid_tim,"solution of LP took %20f seconds for node at level %10d \n",solvetime[i],child[i].level)

								prob.totnodes += 1
							end

						end #for i = nchild:-1:1

						for i = 1:nchild
							if(child[i].UB > -Inf)
								if child[i].UB < prob.GUB
									prob.GUB = child[i].UB
									prob.xUB = child[i].x
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
									@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
									flush(fid_GUB)
								end
							end

							if (child[i].LB > child[i].UB + TOL_GAP)
								println("\nWarning: UB lower than LB!!!")
								@printf(fid_bab,"\nWarning: UB lower than LB!!!")
								println("a.LB = ",child[i].LB," a.UB = ",child[i].UB," GUB = ",prob.GUB)
								#println("a.xUB =", a.xUB)
								@printf(fid_bab,"a.LB = %7.4f, a.UB = %7.4f, GUB = %7.4f\n",child[i].LB,child[i].UB,prob.GUB)
								flush(fid_bab)
							end

							if child[i].LB < prob.GUB*(1 - TOL_GAP)
								BB_form7.insert!(prob,child[i])
								push!(LBs,child[i].LB)
								if child[i].LB < prob.GLB
									println("WARNING !!!! updating GLB ",child[i].LB," ",prob.GLB)
									prob.GLB = child[i].LB
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
								end
							end
						end
						if length(LBs) > 0
							(v,ilb) = findmin(LBs)
							prob.GLB = v
						else
							prob.GLB = prob.GUB
						end
						prob.GAP = max(prob.GUB - prob.GLB,0.0)/max(1.0,abs(prob.GUB))
						for i = 1:nchild
							################################################################
							# DO SOME PRINTING
							################################################################
							#  open    tot          GUB                  father                       child1                        child2
							# 123456 123456 -- 1234567890123 (1234567890123,1234567890123) (1234567890123,1234567890123) (1234567890123,1234567890123)
							################################################################
							@printf(" %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d (%13.6e,%13.6e) %9.3f %4d\n",
								 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,sel_prob.LB,sel_prob.UB,i,child[i].LB,child[i].UB,time()-start_time, num_lb[i])
							@printf(fid_bab," %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d (%13.6e,%13.6e) %9.3f %4d \n",
								 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,sel_prob.LB,sel_prob.UB,i,child[i].LB,child[i].UB,time()-start_time, num_lb[i])
							#flush(fid_bab)
							child[i] = elem_bb_7(1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						if(prob.policy == :sort)
							(BB_form7.sort!)(prob,(>))
						end

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time

			if length(LBs) > 0
				(v,ilb) = findmin(LBs)
				prob.GLB = v
			else
				prob.GLB = prob.GUB
			end
			prob.GAP = max(prob.GUB - prob.GLB,0.0)/max(1.0,abs(prob.GUB))

			return (prob.GUB,prob.GAP,prob.totnodes,prob.numlp,prob.nummilp,execution_time,0.0)
		end
	end

	function solve_7_old(DATA,
					 nloc::Int,
					 MAXNODESin::Number,
					 TOLGAP_IN::Number,
					 time_limit)

		iprint = 0
		#####################################################
		# The main program STARTS HERE
		#####################################################
		# instantiate the Branch & Bound object and define:
		#	management of lp's at B&B nodes
		# 	policy for management of list of open problems;
		#	when the LB for an open problem is computed.
		#	rebuild can take values in {true, false}
		#	policy can take values in {:lifo, :fifo, :sort}
		#	whenlb can take values in {:before, :after}
		#####################################################
		# rebuild  : tells whether LP problems at the B&B tree nodes must be recomputed
		#	     from scratch every time an lb must be computed
		# policy   : defines how the queue is managed. Allowed values are {:lifo, :sort}
		#	     :sort means best-bound visit of B&B tree
		#  		     :lifo means depth-first visit of B&B tree
		# branch   : kind of branch strategy. Allowed values are {:binary, :nary}
		#####################################################

		rebuild  = false
		policy   = :sort
		######################################################
		# NOTE: at the moment branch MUST be :binary
		######################################################
		branch   = :binary

		if(MAXNODESin <= 0)
			println("WARNING: passed value for MAXNODES is <= 0, B&B will explore 0 nodes!")
			MAXNODES = 0
		else
			MAXNODES = MAXNODESin
		end
		if(TOLGAP_IN > 0)
			TOL_GAP = TOLGAP_IN
		end

		prob         = BB_form7.BB_7()
		LBs          = Array{Float64}(undef,0)

		prob.rebuild = rebuild
		prob.policy  = policy
		prob.branch  = branch

		if !(prob.policy in BB_form7.POLICY_VALUES)
			error("ERROR!: possible values for policy are ",transpose(BB_form7.POLICY_VALUES),"\n")
		end
		if !(prob.whenlb in BB_form7.WHENLB_VALUES)
			error("ERROR!: possible values for whenlb are ",transpose(BB_form7.WHENLB_VALUES),"\n")
		end
		if !(prob.branch in BB_form7.BRANCH_VALUES)
			error("ERROR!: possible values for branch are ",transpose(BB_form7.BRANCH_VALUES),"\n")
		end
		if(policy == :sort)
			prob.whenlb = :after
		else
			prob.whenlb = :before
		end

		################ INPUT DATA   #######################
		#####################################################
		# Reading problem data from specified path
		#####################################################
		# problem:
		#
		# min 0.5 x'Qx + c'x + T
		# s.t. Ax <= b, Aeq x = beq
		#      LB <= x <= UB
		#
		LB   = DATA["LB"]
		UB   = DATA["UB"]
		c    = DATA["c"]
		Q    = DATA["Q"]
		A    = DATA["A"]
		b    = DATA["b"]
		Aeq  = DATA["Aeq"]
		beq  = DATA["beq"]
		T    = DATA["T"]
		flag = DATA["flag"]

		#LB,UB,c,Q,A,b,Aeq,beq,T,flag = read_problem(probfile)

		if !(flag)
			println("Some upper or lower bound on the variables is set to infinity. Exiting \n\n")
			return
		end
		(n,~)   = size(Q)
		(m,~)   = size(A)
		(meq,~) = size(Aeq)
		p       = convert(Int,n*(n+1)/2)

		Qvec = zeros(p)
		k = 0
		for j = 1:n
			for i = 1:j
				k += 1
				if i == j
					Qvec[k] = 0.5*Q[i,j]
				else
					Qvec[k] = Q[i,j]
				end
			end
		end

		prob.n   = n
		prob.m   = m
		prob.Q   = Q
		prob.Qvec= Qvec
		prob.c   = c
		prob.T   = T
		prob.A   = A
		prob.Aeq = Aeq
		prob.b   = b
		prob.beq = beq
		################ INPUT DATA  END   ##################

		num_binary = 5

		prob.nvar  = convert(Int,n + p)
		prob.vmap  = Dict("x" => 1:n,
						  "g" => n+1:n+p)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_7, [n for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
						   [Qvec for i in 1:nprocs()],
						   [c for i in 1:nprocs()],
						   [T for i in 1:nprocs()],
						   [A for i in 1:nprocs()],
						   [Aeq for i in 1:nprocs()],
						   [b for i in 1:nprocs()],
						   [beq for i in 1:nprocs()],
						   [prob.vmap for i in 1:nprocs()])
		######## SET DATA FOR PARALLEL PROCESSING END #######

		fid_GUB = open("GUB_stat.txt","w")
		fid_tim = open("tim_stat.txt","w")
		fid_bab = open("bab_stat.txt","w")

		println(fid_GUB, "  node LP's rebuild? ",prob.rebuild)
		println(fid_GUB, "   BB queue policy = ",prob.policy)
		println(fid_GUB, "            whenlb = ",prob.whenlb)
		println(fid_GUB, "branching strategy = ",prob.branch,"\n")

		begin

			start_time = time()
			GAP_time = 0.0
			GAP_node = 0
			if (nloc >0)
				(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,nloc)
				#xstar = [0.25838044394163306, 0.0, 0.0, 0.19120247537184457, 0.0, 0.3433121750327294, 0.0, 0.0, 0.6488588197152324, 0.0, 0.0, 0.0, 0.0, 0.573212917186699, 0.0, 0.0, 0.06177391227639872, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02199876127776128, 0.07204154466034857, 0.1694256448634834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37803804931411833, 0.0, 0.7416802440298111, 0.0, 0.0, 0.9107640307578458, 0.0, 0.0, 0.0, 0.0, 0.23640045712434524, 0.0, 0.0, 0.0, 0.0]
				#bestval = transpose(xstar)*Q*xstar/2 + (transpose(c)*xstar)[1] + T[1]
				prob.GUB = bestval
				prob.xUB = copy(xstar)
			else
				prob.GUB = Inf
			end
			tol           = TOL_GAP
			upperbound1   = prob.GUB
			upperbound    = (1-tol)*prob.GUB
			upperboundmod = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)
			println("build initial model (w/o Qbounds and MC constrs)")
			lp = build_initial_model_7(Q,prob.Qvec,c,T,A,b,Aeq,beq,LB,UB,upperboundmod)
			println("adding Qbounds and MC constrs to the lp")
			add_MC_cons!(lp,n,LB,UB,Q)

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = find_feasible_point(lp,prob)
				show(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				upperbound = (1-tol)*prob.GUB
				upperboundmod = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(xstar)
				lp = build_model_7(Q,prob.Qvec,c,T,A,b,Aeq,beq,LB,UB,upperboundmod)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end

			e = elem_bb_7(n)
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,lowx,upx,iprint)
			println("    LB = ",lb,"     GUB = ",prob.GUB)
			if iprint >= 1
				println("                  GUB = ",prob.GUB)
				println("                   UB = ",e.UB)
			end
			(lb, ub, solvetime) = compute_lowerbound!(e,prob,tol,iprint,false)
			println("    LB = ",lb,"     GUB = ",prob.GUB)
			if iprint >= 1
				println("                  GUB = ",prob.GUB)
				println("                   UB = ",e.UB)
				println("lb at root node = ", lb,      "   e.UB = ",e.UB)
				println("            e.x = ",e.x)
			end

			@printf(fid_bab," lb at root node is %13.6e\n",lb)
			flush(fid_bab)
			@printf(fid_tim,"solution of LP took %20f seconds for node at level %10d \n",solvetime,e.level)
			flush(fid_tim)

			@printf(" lb at root node is %13.6e\n",lb)
			@printf(" ub at root node is %13.6e\n",prob.GUB)

			prob.totnodes += 1

			if (e.LB == Inf)
				execution_time = time()-start_time

				#write_final_stat(fid_STA,fid_GUB,prob,[],execution_time)

				if iprint > 2
					println(" xUB = ",prob.xUB)
				end

				close(fid_GUB)
				close(fid_tim)
				close(fid_bab)

				return (prob.GUB,0.0,prob.totnodes,prob.numlp,prob.nummilp,execution_time,0.0)

			end

			prob.GLB = e.LB
			if e.UB < prob.GUB
				prob.GUB = e.UB
				prob.xUB = copy(e.x)
				@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
				flush(fid_GUB)
			end

			push!(prob.open_probs,e)
			push!(LBs,e.LB)
			prob.nnodes += 1

			if(prob.policy == :sort)
				(BB_form7.sort!)(prob,(>))
			end
			prob.GLB = minimum(LBs)
			prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))

			@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
			flush(fid_GUB)

			@printf(" open    tot          GUB          GAP             father info           ch.#           child info\n")
			@printf("                                         (     LB              UB    )        (     LB              UB    )\n")
			@printf(fid_bab," open    tot          GUB          GAP             father info           ch.#           child info\n")
			@printf(fid_bab,"                                         (     LB              UB    )        (     LB              UB    )\n")

			@printf(" %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d \n",
				 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,e.LB,e.UB,0)
			@printf(fid_bab," %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d \n",
				 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,e.LB,e.UB,0)
			flush(fid_bab)

			while (prob.nnodes > 0) && (prob.totnodes < MAXNODES) && (time()-start_time <= time_limit)
				sel_prob = BB_form7.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/max(1.0,abs(prob.GUB)) >=  -TOL_GAP)
					println("Fathom sel_prob since ", sel_prob.LB, " >= ", prob.GUB*(1 -TOL_GAP))
					continue
				end
				begin

					local child = subdivide(prob,sel_prob,iprint)
					nchild = length(child)
					index = sel_prob.index
					if iprint >= 1
						@printf("Selected  node = %8d, Lower bound = %13.6e, GUB = %13.6e\n",i,sel_prob.LB,prob.GUB)
						@printf("Percentage GAP = %8.6f, Elapsed tim = %13.6e, numnodes = %6d, open nodes = %6d\n",prob.GAP,time()-start_time,prob.totnodes,prob.nnodes)
						@printf("Branching  var = %8d,      x value= %13.6e, xmn = %13.6e, xmx = %13.6e\n",
								sel_prob.index,sel_prob.x[index],sel_prob.xmin[index],sel_prob.xmax[index])
					end
					if iprint > 2
						print("hit RETURN to continue ...")
						readline()
					end

					if nchild > 0
						solvetime = [0.0;0.0]
						lb = [0.0; 0.0]
						ub = [0.0; 0.0]
						num_lb = [0; 0]
						#Threads.@threads for i = 1:nchild
						#println(Threads.threadid())
						for i = 1:nchild
							#a = child[i]
							child[i].level = (sel_prob.level)+1

							if (child[i].LB < Inf)
								(lb[i], ub[i], solvetime[i]) = compute_lowerbound!(child[i],prob,tol,iprint,false)
								@printf(fid_tim,"solution of LP took %20f seconds for node at level %10d \n",solvetime[i],child[i].level)

								prob.totnodes += 1
							end

						end #for i = nchild:-1:1

						for i = 1:nchild
							if(child[i].UB > -Inf)
								if child[i].UB < prob.GUB
									prob.GUB = child[i].UB
									prob.xUB = child[i].x
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
									@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
									flush(fid_GUB)
								end
							end

							if (child[i].LB > child[i].UB + TOL_GAP)
								println("\nWarning: UB lower than LB!!!")
								@printf(fid_bab,"\nWarning: UB lower than LB!!!")
								println("a.LB = ",child[i].LB," a.UB = ",child[i].UB," GUB = ",prob.GUB)
								#println("a.xUB =", a.xUB)
								@printf(fid_bab,"a.LB = %7.4f, a.UB = %7.4f, GUB = %7.4f\n",child[i].LB,child[i].UB,prob.GUB)
								flush(fid_bab)
							end

							if child[i].LB < prob.GUB*(1 - TOL_GAP)
								BB_form7.insert!(prob,child[i])
								push!(LBs,child[i].LB)
								if child[i].LB < prob.GLB
									println("WARNING !!!! updating GLB ",child[i].LB," ",prob.GLB)
									prob.GLB = child[i].LB
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
								end
							end
						end
						if length(LBs) > 0
							(v,ilb) = findmin(LBs)
							prob.GLB = v
						else
							prob.GLB = prob.GUB
						end
						prob.GAP = max(prob.GUB - prob.GLB,0.0)/max(1.0,abs(prob.GUB))
						for i = 1:nchild
							################################################################
							# DO SOME PRINTING
							################################################################
							#  open    tot          GUB                  father                       child1                        child2
							# 123456 123456 -- 1234567890123 (1234567890123,1234567890123) (1234567890123,1234567890123) (1234567890123,1234567890123)
							################################################################
							@printf(" %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d (%13.6e,%13.6e) %9.3f %4d\n",
								 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,sel_prob.LB,sel_prob.UB,i,child[i].LB,child[i].UB,time()-start_time, num_lb[i])
							@printf(fid_bab," %6d %6d -- %13.6e %7.4f%% (%13.6e,%13.6e) %6d (%13.6e,%13.6e) %9.3f %4d \n",
								 prob.nnodes,prob.totnodes,prob.GUB,100*prob.GAP,sel_prob.LB,sel_prob.UB,i,child[i].LB,child[i].UB,time()-start_time, num_lb[i])
							#flush(fid_bab)
							child[i] = elem_bb_7(1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						if(prob.policy == :sort)
							(BB_form7.sort!)(prob,(>))
						end

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time
			return (prob.GUB,prob.GAP,prob.totnodes,prob.numlp,prob.nummilp,execution_time,0.0)
		end
	end

	########################u#############################
	# given an elem_bb_7 e, for which the lower bound
	# has already been compute, subdivide it into 2
	# subproblems using rules derived from the KKT conditions
	#####################################################
	function subdivide(prob,e::elem_bb_7,iprint)

		n             = prob.n
		m             = prob.m
		meq           = prob.meq
		A             = prob.A
		Aeq           = prob.Aeq
		b             = prob.b
		beq           = prob.beq
		Q             = prob.Q
		c             = prob.c
		T 			  = prob.T
		vmap		  = prob.vmap

		imaxgap_x = e.index
		#println(e.index)

		gap       = zeros(n)
		for j = 1:n
			gap[j] = 0.0
			for i = 1:n
				if i <= j
					k = convert(Int,(j-1)*j/2) + i
				else
					k = convert(Int,(i-1)*i/2) + j
				end
				if i==j
					gap[j] += 0.5*Q[i,j]*(e.x[i]*e.x[j] - e.g[k])
				else
					gap[j] += Q[i,j]*(e.x[i]*e.x[j] - e.g[k])
				end
				#if (j==e.index)
				#	println("i=",i," g=",e.g[k]," Q=",Q[i,j]," gap=",e.x[i]*e.x[j] - e.g[k])
				#end
			end
		end
		maxgap_x  = gap[imaxgap_x]

		if iprint >= 1
			println("branching on ", imaxgap_x," the gap is ", maxgap_x)
			#println("gap: ",gap)
		end

		if maxgap_x <= 1.e-6
			println("WARNING from subdivide: no further branching is possible!!")
			return Array{elem_bb_7,1}(undef,0)
		end
		#NOTE: branch is forcedly binary!
		nT = 2
		###################################
		# builds the 2 subproblems
		###################################
		eT = Array{elem_bb_7}(undef,nT)
		for iT = 1:nT
			eT[iT] = elem_bb_7(n)
			eT[iT].LB = e.LB
			eT[iT].UB = e.UB
			eT[iT].xUB= copy(e.xUB)
			eT[iT].x  = copy(e.x)
			eT[iT].g  = copy(e.g)
			eT[iT].xmin  = copy(e.xmin)
			eT[iT].xmax  = copy(e.xmax)
			if iT == 2
				eT[iT].xmin[imaxgap_x] = e.x[imaxgap_x]
			else
				eT[iT].xmax[imaxgap_x] = e.x[imaxgap_x]
			end #end if iT == 1

		end #end for iT

		return eT
	end

	function compute_lowerbound!(e::elem_bb_7,prob, tol, iprint, isroot)
		#questo bound è usato solo ai nodi INTERMEDI
		#######################################
		# BT = 0  NESSUN BOUND TIGHTENING
		# BT = 1  1 GIRO DI BOUND TIGHTENING
		# BT = Inf  STRONG BOUND TIGHTENING
		#######################################
		BT            = Inf

		n             = prob.n
		m             = prob.m
		meq           = prob.meq
		A             = prob.A
		Aeq           = prob.Aeq
		b             = prob.b
		beq           = prob.beq
		Q             = prob.Q
		Qvec          = prob.Qvec
		c             = prob.c
		T 			  = prob.T
		GUB           = prob.GUB
		lowx          = e.xmin
		upx           = e.xmax
		wwaux         = e.x
		ttaux         = e.g
		vmap          = prob.vmap
		upperbound    = (1-tol)*GUB
		upperboundmod = (1-tol)*(GUB-T[1])
		feas          = true
		num_lb        = 0
		solvelp       = 0.2*n
		lowerbound    = e.LB
		originallower = lowerbound
		lowerboundold = originallower-1

		start_time =time()
		gap = zeros(n)

		while true
			solvelp = 0
			totlp   = 0
			totlp1  = 0
			for j = 1:n
				gap[j] = 0.0
				for i = 1:n
					if i <= j
						k = convert(Int,(j-1)*j/2) + i
					else
						k = convert(Int,(i-1)*i/2) + j
					end
					if (i==j)
						gap[j] += 0.5*Q[i,j]*(wwaux[i]*wwaux[j] - ttaux[k])
					else
						gap[j] += Q[i,j]*(wwaux[i]*wwaux[j] - ttaux[k])
					end
				end
				if (gap[j] > 0)
					solvelp=solvelp+1;
				end
			end

			#if (lowerboundold<originallower)
			#	totlp=min(n,2*solvelp)
			#	totlp1=10
			#else
			#	totlp=max(solvelp,0.2*n)
			#	totlp1=10
			#end
			#totlp  = convert(Int,totlp)
			#totlp1 = convert(Int,totlp1)
			totlp1 = solvelp
			p      = sortperm(gap,rev=true)
			ind    = p

			if BT >= 1
				#####################################
				# RISOLVO totlp + 2*totlp1 problemi
				#####################################
				lowx0 = copy(lowx)
				upx0  = copy(upx)
				pmap(set_lp_7,[lowx0 for k in 1:nprocs()],
							[upx0 for k in 1:nprocs()],
							[upperboundmod for k in 1:nprocs()])
				resp1   = pmap(solutore_max_x_7, [ind[k] for k in 1:totlp1])
				resp2   = pmap(solutore_min_x_7, [ind[k] for k in 1:totlp1])
				for k = 1:totlp1
					prob.numlp = prob.numlp+2
					i = ind[k]
					upx[i]  = resp1[k][1] #mx_i
					lowx[i] = resp2[k][1] #mn_i
					feas   = resp2[k][2]
				end#end for i=1:n
				pmap(clear_lp_7, [k for k in 1:nprocs()])
			else
				feas = true
			end
			
			if feas

				lp = build_model_7(Q,Qvec,c,T,A,b,Aeq,beq,lowx,upx,upperboundmod)

				add_Xmax_cons!(lp,n,upx)
				add_MC_cons!(lp,n,lowx,upx,Q)

				lowerboundold = lowerbound
				############################################
				# CALCOLO il LOWERBOUND
				############################################
				(lowerbound,sol,feas) = compute_low(lp,prob)
				if feas
					x      = sol[vmap["x"]]
					g      = sol[vmap["g"]]
					wwaux  = x
					ttaux  = g
					e.LB   = lowerbound
					e.x    = copy(x)
					e.g    = copy(g)
					e.xmin = lowx
					e.xmax = upx
					e.index= 1
					UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
					if iprint >= 1
						println("lowold = ",lowerboundold,"   low = ",lowerbound)
					end
					if UB < GUB
						GUB = UB
						upperb = (1-tol)*UB
						upperbound    = (1-tol)*GUB
						upperboundmod = (1-tol)*(GUB-T[1])
						if iprint >= 1
							println("upperb = ",GUB)
						end
					end
					e.LB   = lowerbound

					if UB < e.UB
						e.UB  = UB
						e.xUB = copy(x)
					end
					if (UB < prob.GUB)
						prob.GUB = UB
						prob.xUB = x
						GUB      = prob.GUB
					end
					Gurobi.free_model(lp)
				else
					break
					Gurobi.free_model(lp)
				end
			else
				break
			end
			if iprint >= 1
				@printf(" solvelp = %3d totlp = %3d totlp1 = %3d lowerbound: %15.8e\n",solvelp,totlp,totlp1,lowerbound);
				@printf("criterio: lhs = %15.8e  rhs = %15.8e\n",(lowerbound - lowerboundold),max(0.5*tol,0.01*(GUB-lowerbound)))
			end
			if ((lowerbound - lowerboundold) <= max(0.5*tol,0.01*(GUB-lowerbound)))
				break
			end
			if isroot 
				break
			end
			if BT <= 1
				break
			end
		end #while
		solvetime = time()-start_time

		if (~feas)
			#println("LB not feasible")
			e.LB = Inf
			e.UB = Inf
			lowerbound = Inf
		else
			gap       = zeros(n)
			for j = 1:n
				gap[j] = 0.0
				for i = 1:n
					if i <= j
						k = convert(Int,(j-1)*j/2) + i
					else
						k = convert(Int,(i-1)*i/2) + j
					end
					if i==j
						gap[j] += 0.5*Q[i,j]*(wwaux[i]*wwaux[j] - ttaux[k])
					else
						gap[j] += Q[i,j]*(wwaux[i]*wwaux[j] - ttaux[k])
					end
				end
			end
			p      = sortperm(gap,rev=true)
			e.index= p[1]
		end
		return (lowerbound, e.UB, solvetime)

	end

	function add_Xmax_cons!(lp,n,upx)
		Gurobi.set_dblattrarray!(lp,"UB",1,n,upx)
		Gurobi.update_model!(lp)
	end

	function compute_strong_McCormick_bound!(e::elem_bb_7,prob,tol,lowx,upx,iprint)
		n       = prob.n
		p       = convert(Int,n*(n+1)/2)
		m       = prob.m
		meq		= prob.meq
		A       = prob.A
		Aeq     = prob.Aeq
		b       = prob.b
		beq     = prob.beq
		Q       = prob.Q
		Qvec	= prob.Qvec
		c   	= prob.c
		T 		= prob.T
		GUB     = prob.GUB
		vmap    = prob.vmap
		lowold  = -Inf
		lowerb  = 0.0
		upperb  = (1.0-tol)*GUB
		upperbmod  = (1.0-tol)*(GUB-T[1])
		feas    = true
		ww      = zeros(n)
		tt      = zeros(p)
		solvelp = 0.2*n

		start_time = time()

		#while ((lowerb - lowold) > max(tol,0.1*tol*GUB))
		while true
			if iprint >= 1
				@printf("lowerb-lowold = %15.8e -- max(tol,0.1*tol*GUB) = %15.8e\n",(lowerb - lowold),max(tol,0.1*tol*GUB))
			end

			lp  = build_model_7(Q,Qvec,c,T,A,b,Aeq,beq,lowx,upx,upperbmod)

			gap     = ones(n)
			if (lowold < 0)
				gap     = ones(n)
				totlp   = n
			else
				totlp = 0
				for j = 1:n
					gap[j] = 0.0
					for i = 1:n
						if i <= j
							k = convert(Int,(j-1)*j/2) + i
						else
							k = convert(Int,(i-1)*i/2) + j
						end
						if i==j
							gap[j] += 0.5*Q[i,j]*(ww[i]*ww[j] - tt[k])
						else
							gap[j] += Q[i,j]*(ww[i]*ww[j] - tt[k])
						end
					end
					if gap[j] > 0
						totlp += 1
					end
				end
				totlp   = min(solvelp,totlp)
			end

			num_LP = convert(Int,totlp)
			p      = sortperm(gap,rev=true)
			ind    = p[1:num_LP]
			#println(gap)
			#println(ind)
			#println(num_LP)

			###############################
			# RISOLVO 3*num_LP problemi
			###############################
			for k = 1:num_LP
				i = ind[k]
				(mn_i,  feas) = compute_min_x(lp,prob,i)
				(mx_i,  feas) = compute_max_x(lp,prob,i)
				upx[i] = mx_i
				lowx[i]= mn_i
				if iprint >= 1
					@printf(" lowx[%3d] = %13.6e upx[%3d] = %13.6e\n",i,lowx[i],i,upx[i])
				end
			end#end for i=1:n

			if (feas)
				add_MC_cons!(lp,n,lowx,upx,Q)
				add_GUB_con!(lp,n,Qvec,c,upperbmod)
				lowold = lowerb
				############################################
				# CALCOLO il LOWERBOUND
				############################################
				(lowerb,sol,feas) = compute_low(lp,prob)
				if feas
					x      = sol[vmap["x"]]
					g      = sol[vmap["g"]]
					ww     = x
					tt     = g
					e.LB   = lowerb
					e.x    = copy(x)
					e.g    = copy(g)
					e.xmin = lowx
					e.xmax = upx
					e.index= 1
					if iprint >= 1
						println("lowold = ",lowold,"   low = ",lowerb)
					end
					UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
					if UB < GUB
						GUB = UB
						upperb = (1-tol)*UB
						upperbmod = (1-tol)*(UB-T[1])
					end

					if UB < e.UB
						e.UB  = UB
						e.xUB = copy(x)
					end
					if (UB < prob.GUB)
						prob.GUB = UB
						prob.xUB = copy(x)
						GUB      = prob.GUB
					end
				else
					break
					Gurobi.free_model(lp)
				end
			else
				break
				Gurobi.free_model(lp)
			end
			if iprint >= 1
				@printf("                      lowerbound:%15.8e\n",lowerb);
			end
			if ((lowerb - lowold) <= max(0.5*tol,0.01*(GUB-lowerb)))
				break
			end
			break
		end
		solvetime = time()-start_time
		if (~feas)
			println("LB not feasible")
			e.LB = Inf
			e.UB = Inf
			lowerb = Inf
		end
		return (lowerb, e.UB, solvetime)

	end

	function compute_min_l(lp,prob,i)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:minimize)
		coefs = zeros(prob.nvar)
		coefs[collect(prob.vmap["x"])] = prob.Q[i,:]
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
		optimize(lp)
		prob.numlp = prob.numlp+1
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mn_li = get_objval(lp)
			feas  = true
		else
			#println("\t infeasibility detected computing mn_li for i =",i)
			mn_li = 0.0
			feas  = false
		end
		return mn_li, feas
	end

	function compute_max_l(lp,prob,i)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:maximize)
		coefs = zeros(prob.nvar)
		coefs[collect(prob.vmap["x"])] = prob.Q[i,:]
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
		optimize(lp)
		prob.numlp = prob.numlp+1
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mx_li = get_objval(lp)
			feas  = true
		else
			#println("\t infeasibility detected computing mx_li for i =",i)
			mx_li = 0.0
			feas  = false
		end
		return mx_li, feas
	end

	function compute_max_x(lp,prob,i)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:maximize)
		coefs = zeros(prob.nvar)
		coefs[collect(prob.vmap["x"])[i]] = 1.0
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
		optimize(lp)
		prob.numlp = prob.numlp+1
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mx_i = get_objval(lp)
			feas = true
		else
			#println("\t infeasibility detected computing mx_i for i =",i)
			mx_i = 0.0
			feas = false
		end
		return mx_i, feas
	end

	function compute_min_x(lp,prob,i)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:minimize)
		coefs = zeros(prob.nvar)
		coefs[collect(prob.vmap["x"])[i]] = 1.0
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
		optimize(lp)
		prob.numlp = prob.numlp+1
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mn_i = get_objval(lp)
			feas = true
		else
			#println("\t infeasibility detected computing mx_i for i =",i)
			mn_i = 0.0
			feas = false
		end
		return mn_i, feas
	end
	function compute_low(lp,prob)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:minimize)
		coefs = zeros(prob.nvar)
		coefs[prob.vmap["x"]] = prob.c
		coefs[prob.vmap["g"]] = prob.Qvec
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
		optimize(lp)
		prob.numlp = prob.numlp+1
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			low = get_objval(lp) + prob.T[1]
			sol = get_solution(lp)
			feas = true
		else
			#println("\t infeasibility detected computing lower bound")
			low  = Inf
			feas = false
			sol  = []
		end
		return low, sol, feas

	end

	function find_feasible_point(lp,prob)
			Gurobi.set_sense!(lp,:minimize)
			coefs = zeros(prob.nvar)
			Gurobi.set_objcoeffs!(lp,coefs)
			Gurobi.update_model!(lp)
			optimize(lp)
			prob.numlp = prob.numlp+1
			feas = true
			if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
				sol = get_solution(lp)
				feas  = true
			else
				#println("\t infeasibility detected at root node !!!!!\n\n")
				sol = zeros(prob.nvar)
				feas  = false
			end
			return (reshape(sol[1:prob.n],(prob.n,1)),feas)
	end

	function add_variables!(lp,n,lowx,upx)
		#println("\t\t\tAdding x variables ...")
		add_cvars!(lp,zeros(n),reshape(lowx,(n,)),reshape(upx,(n,)))
		#println("\t\t\tAdding g variables ...")
		p = convert(Int,n*(n+1)/2)
		add_cvars!(lp,zeros(p),zeros(p),ones(p))
		#println("\t\t\t... done")
		update_model!(lp)
	end
	function add_standard_cons!(lp,n,lowx,upx,A,b,Aeq,beq)
		####################################################
		# Questa proc. aggiunge i vincoli:
		# A*x <= b, Aeq*x = beq
		####################################################
		Gurobi.set_dblattrarray!(lp,"UB",1,n,reshape(upx,(n,)))
		Gurobi.set_dblattrarray!(lp,"LB",1,n,reshape(lowx,(n,)))
		#println("\t\t\tAdding constraints b_j'x <= v ...")
		m,~   = size(A)
		meq,~ = size(Aeq)
		for j = 1:m
			add_constr!(lp,collect(1:n),A[j,:],'<',b[j])
		end
		for j = 1:meq
			add_constr!(lp,collect(1:n),Aeq[j,:],'=',beq[j])
		end
		update_model!(lp)
	end
	function add_GUB_con!(lp,n,Qvec,c,upperboundmod)
		####################################################
		# Questa proc. aggiunge il vincolo:
		# e'*g + c'x + const_ <= GUB
		####################################################
		#println("\t\t\tAdding constraint e'*g + c'*x <= GUB")
		#println("add_GUB_con!: ",const_)
		#SISTEMARE !!!!!!!
		p       = convert(Int,n*(n+1)/2)
		add_constr!(lp,[collect(1:n); collect((n+1):(n+p))],
				   [reshape(c,(n,));         Qvec],'<',upperboundmod)
		update_model!(lp)
	end
	function add_MC_cons!(lp,n,lowx,upx,Q)
		####################################################
		# Questa proc. aggiunge i vincoli: (se Qij > 0)
		#   g_k = Xij ≥ lowx[i]x[j]+lowx[j]x[i] - lowx[i]lowx[j]
		#   g_k = Xij ≥ upx[i]x[j]+upx[j]x[i] − upx[i]upx[j]
		#
		# (se Qij < 0)
		#   g_k = Xij <= lowx[i]x[j] + upx[j]x[i] -lowx[i]upx[j]
		#   g_k = Xij <= upx[i]x[j] + lowx[j]x[i] -upx[i]lowx[j]
		####################################################
		k = 0
		p = convert(Int, n*(n+1)/2)
		for j = 1:n
			for i = 1:j
				k += 1
				coefx = zeros(n)
				coefg = zeros(p)
				coefg[k] = 1.0
				if Q[i,j] > 0
					coefx[i] = -lowx[j]
					if i==j
						coefx[j] += -lowx[i]
						#println("MC1: coefx=",coefx[j]," coefg=",coefg[k],">=",-lowx[i]*lowx[j])
					else
						coefx[j] = -lowx[i]
					end
					add_constr!(lp,[collect(1:n);collect((n+1):(n+p))],[coefx;coefg],'>',-lowx[i]*lowx[j])
					coefx[i] = -upx[j]
					if i==j
						coefx[j] += -upx[i]
						#println("MC2: coefx=",coefx[j]," coefg=",coefg[k],">=",-upx[i]*upx[j])
					else
						coefx[j] = -upx[i]
					end
					add_constr!(lp,[collect(1:n);collect((n+1):(n+p))],[coefx;coefg],'>',-upx[i]*upx[j])
				elseif Q[i,j] < 0
					coefx[i] = -upx[j]
					if i==j
						coefx[j] += -lowx[i]
					else
						coefx[j] = -lowx[i]
					end
					add_constr!(lp,[collect(1:n);collect((n+1):(n+p))],[coefx;coefg],'<',-lowx[i]*upx[j])
					coefx[i] = -lowx[j]
					if i==j
						coefx[j] += -upx[i]
					else
						coefx[j] = -upx[i]
					end
					add_constr!(lp,[collect(1:n);collect((n+1):(n+p))],[coefx;coefg],'<',-upx[i]*lowx[j])
				end
			end
		end
		update_model!(lp)
	end


	#####################################################
	# This function finds an initial GUB by doing
	# ntrials minimizations with IPOPT from random
	# starting points
	#####################################################
	function find_initial_GUB(S,c,T,A,b,Aeq,beq,ntrials)
		############################################
		#
		# Il problema è:
		#
		# min 0.5 x'Sx + c'x + T
		# s.t. 0 <= x <= 1,
		#      Ax <= b, Aeq x = beq
		#
		############################################

		println("\n\n------------------------------------------------\n")
		println(    " Try to heuristically determine glob. solution")
		println(    " via multistart IPOPT optimization")
		println(    "------------------------------------------------\n")

		(n,  ~) = size(S)
		(m,  ~) = size(A)
		(meq,~) = size(Aeq)

		function eval_g(x, g)
			################################
			# g[1:m] = A*x - b
			# g[m+1:m+meq] = Aeq*x - beq
			################################
			g[1:m] = A*x - b
			g[m+1:m+meq] = Aeq*x - beq
		end
		function eval_jac_g(x, mode, rows, cols, values)
		  if mode == :Structure
			h = 1
			for j = 1:m
				for i = 1:n
					rows[h] = j; cols[h] = i
					h += 1
				end
			end
			for j = 1:meq
				for i = 1:n
					rows[h] = m+j; cols[h] = i
					h += 1
				end
			end
		  else
			h = 1
			for j = 1:m
				for i = 1:n
					values[h] = A[j,i]
					h += 1
				end
			end
			for j = 1:meq
				for i = 1:n
					values[h] = Aeq[j,i]
					h += 1
				end
			end
		  end
		end
		function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
		end
		function eval_f(x)
			ret = transpose(x)*S*x/2 + (transpose(c)*x)[1] + T[1]
			return ret
		end
		function eval_grad_f(x, grad_f)
			ret = S*x + c
			for i = 1:n
				grad_f[i] = ret[i]
			end
		end

		g_L = -Inf*ones(m+meq)
		g_U = zeros(m+meq)
		g_L[m+1:m+meq] = zeros(meq)
		x_L = zeros(n)
		x_U = ones(n)
		IPprob = createProblem(convert(Int,n),x_L,x_U,convert(Int,m+meq),g_L,g_U,convert(Int,n*m+n*meq),0,eval_f,eval_g,eval_grad_f,eval_jac_g,eval_h)
		addOption(IPprob,"hessian_approximation","limited-memory")
		addOption(IPprob,"honor_original_bounds","yes")
		addOption(IPprob,"print_level",0)
		xstar = zeros(n)
		bestval = +Inf
		global bestval
		for iter = 1:ntrials
			coefs = rand(n)
			#println(coefs)
			IPprob.x = coefs
			status = solveProblem(IPprob)
			fstar = eval_f(IPprob.x)
			#fstar = IPprob.obj_val
			println("solution is ",fstar)
			if (fstar < bestval)
				bestval = fstar
				xstar = copy(IPprob.x)
				#println("current best solution is ",bestval)
				#println(xstar)
			end
		end
		return (bestval,reshape(xstar,(n,1)))
	end
end
