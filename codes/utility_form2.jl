__precompile__()

module utility_form2
	using Distributed
	using BB_form2
	@everywhere using partools_form2
	using Gurobi
	using Ipopt
	using MAT
	using LinearAlgebra
	using Printf
	using Random

	export compute_rootnode_gap_2
	export solve_2
	export solve_2_old

	function compute_rootnode_gap_2(DATA,
					 nloc::Int,
					 MAXNODESin::Number,
					 TOLGAP_IN::Number,
					 time_limit)
		iprint = 0

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

		prob         = BB_form2.BB_2()
		LBs          = Array{Float64}(undef,0)

		prob.rebuild = rebuild
		prob.policy  = policy
		prob.branch  = branch

		if !(prob.policy in BB_form2.POLICY_VALUES)
			error("ERROR!: possible values for policy are ",transpose(BB_form2.POLICY_VALUES),"\n")
		end
		if !(prob.whenlb in BB_form2.WHENLB_VALUES)
			error("ERROR!: possible values for whenlb are ",transpose(BB_form2.WHENLB_VALUES),"\n")
		end
		if !(prob.branch in BB_form2.BRANCH_VALUES)
			error("ERROR!: possible values for branch are ",transpose(BB_form2.BRANCH_VALUES),"\n")
		end
		if(policy == :sort)
			prob.whenlb = :after
		else
			prob.whenlb = :before
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
			solved = false
			error  = true
			return 0.0, Inf, prob, LBs, solved, error
		end
		(n,~)   = size(Q)
		(m,~)   = size(A)
		(meq,~) = size(Aeq)

		prob.n   = n
		prob.m   = m
		prob.Q   = Q
		prob.c   = c
		prob.T   = T
		prob.A   = A
		prob.Aeq = Aeq
		prob.b   = b
		prob.beq = beq
		################ INPUT DATA  END   ##################

		prob.nvar  = 2n
		prob.vmap  = Dict("x" => 1:n,
						  "g" => n+1:2n)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_2, [n for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
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

		begin

			start_time = time()
			if (nloc >0)
				(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,nloc)
				prob.GUB = bestval
				prob.xUB = copy(xstar)
			else
				prob.GUB = Inf
			end

			tol           = TOL_GAP
			upperbound1   = prob.GUB
			upperbound    = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)

			println("build initial model (w/o Qbounds and MC constrs)")
			lp = build_initial_model_2(Q,c,T,A,b,Aeq,beq,LB,UB,upperbound)
			println("computing low and up for Qbounds")
			for i=1:n
				(mn_li, feas) = compute_min_l(lp,prob,i)
				(mx_li, feas) = compute_max_l(lp,prob,i)
				low[i]  = mn_li
				up[i]   = mx_li
			end
			println("adding Qbounds and MC constrs to the lp")
			add_Qbound_cons_2!(lp,n,low,up,Q)
			add_MC_cons_2!(lp,n,low,up,UB,Q)

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = find_feasible_point(lp,prob)
				show(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				upperbound = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(xstar)
				Gurobi.free_model(lp)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end

			e = elem_bb_2(n)
			for i = 1:n
				e.lmax[i] = low[i] #maximum(prob.Q[i,:])
				e.lmin[i] = up[i]  #minimum(prob.Q[i,:])
			end
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,low,up,lowx,upx,iprint)
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

			@printf(" lb at root node is %13.6e\n",lb)
			@printf("solution of LP took %20f seconds for node at level %10d \n",solvetime,e.level)

			@printf(" lb  at root node is %13.6e\n",lb)
			@printf(" ub  at root node is %13.6e\n",prob.GUB)
			@printf(" gap at root node is %13.6e\n",prob.GAP)

			prob.totnodes += 1

			if (e.LB == Inf)
				execution_time = time()-start_time

				#write_final_stat(fid_STA,fid_GUB,prob,[],execution_time)

				if iprint >= 0
					println(" xUB = ",prob.xUB)
				end

				close(fid_GUB)
				close(fid_tim)
				close(fid_bab)

				solved = true
				error  = false
				return rootgap_time, 0.0, prob, LBs, solved, error

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
				(BB_form2.sort!)(prob,(>))
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

			solved = false
			error  = false
			return rootgap_time, prob.GAP, prob, LBs, solved, error
		end
	end

	function solve_2(DATA,prob,LBs,
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
			upperbound    = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)

			#qui si potrebbe uscire per restituire il gap al root

			while (prob.nnodes > 0) && (prob.totnodes < MAXNODES) && (time()-start_time <= time_limit)
				sel_prob = BB_form2.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/abs(prob.GUB) >=  -TOL_GAP)
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
						@printf("                                q value= %13.6e, qmn = %13.6e, qmx = %13.6e\n",
								transpose(S[index,:])*sel_prob.x,sel_prob.lmin[index],sel_prob.lmax[index])
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
								BB_form2.insert!(prob,child[i])
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
							#println("(*) LBs = ",LBs)
							#println("(*) lb = ",lb," ub = ",ub)
							#println("(*) GLB = ",prob.GLB," GUB = ",prob.GUB," GAP = ",prob.GAP)
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
							child[i] = elem_bb_2(1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						if(prob.policy == :sort)
							(BB_form2.sort!)(prob,(>))
						end

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time
			println(prob.xUB)

			close(fid_tim)
			close(fid_GUB)
			close(fid_bab)

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

	function solve_2_old(DATA,
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

		prob         = BB_form2.BB_2()
		LBs          = Array{Float64}(undef,0)

		prob.rebuild = rebuild
		prob.policy  = policy
		prob.branch  = branch

		if !(prob.policy in BB_form2.POLICY_VALUES)
			error("ERROR!: possible values for policy are ",transpose(BB_form2.POLICY_VALUES),"\n")
		end
		if !(prob.whenlb in BB_form2.WHENLB_VALUES)
			error("ERROR!: possible values for whenlb are ",transpose(BB_form2.WHENLB_VALUES),"\n")
		end
		if !(prob.branch in BB_form2.BRANCH_VALUES)
			error("ERROR!: possible values for branch are ",transpose(BB_form2.BRANCH_VALUES),"\n")
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

		prob.n   = n
		prob.m   = m
		prob.Q   = Q
		prob.c   = c
		prob.T   = T
		prob.A   = A
		prob.Aeq = Aeq
		prob.b   = b
		prob.beq = beq
		################ INPUT DATA  END   ##################

		num_binary = 5
		prob.nvar  = 2n
		prob.vmap  = Dict("x" => 1:n,
						  "g" => n+1:2n)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_2, [n for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
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
				prob.GUB = bestval
				prob.xUB = copy(xstar)
			else
				prob.GUB = Inf
			end
			tol           = TOL_GAP
			upperbound1   = prob.GUB
			upperbound    = (1-tol)*(prob.GUB-T[1])
			lowerboundold = -1000000.0
			lowerbound    =        0.0
			low           = zeros(n)
			up            = zeros(n)
			lowx          = zeros(n)
			upx           = ones(n)
			ww            = zeros(n)
			tt            = zeros(n)
			println("build initial model (w/o Qbounds and MC constrs)")
			lp = build_initial_model_2(Q,c,T,A,b,Aeq,beq,LB,UB,upperbound)
			println("computing low and up for Qbounds")
			for i=1:n
				(mn_li, feas) = compute_min_l(lp,prob,i)
				(mx_li, feas) = compute_max_l(lp,prob,i)
				low[i]  = mn_li
				up[i]   = mx_li
			end
			println("adding Qbounds and MC constrs to the lp")
			add_Qbound_cons_2!(lp,n,low,up,Q)
			add_MC_cons_2!(lp,n,low,up,UB,Q)

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = find_feasible_point(lp,prob)
				show(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				upperbound = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(xstar)
				Gurobi.free_model(lp)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end

			e = elem_bb_2(n)
			for i = 1:n
				e.lmax[i] = maximum(prob.Q[i,:])
				e.lmin[i] = minimum(prob.Q[i,:])
			end
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,low,up,lowx,upx,iprint)
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

			rootgap_time = time()-start_time
			prob.GAP = (prob.GUB - lb)/max(1.0,abs(prob.GUB))

			@printf(fid_bab," lb at root node is %13.6e\n",lb)
			flush(fid_bab)
			@printf(fid_tim,"solution of LP took %20f seconds for node at level %10d \n",solvetime,e.level)
			flush(fid_tim)

			@printf(" lb  at root node is %13.6e\n",lb)
			@printf(" ub  at root node is %13.6e\n",prob.GUB)
			@printf(" gap at root node is %13.6e\n",prob.GAP)

			#qui si potrebbe uscire per restituire il gap al root

			prob.totnodes += 1

			if (e.LB == Inf)
				execution_time = time()-start_time

				#write_final_stat(fid_STA,fid_GUB,prob,[],execution_time)

				if iprint >= 0
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
				(BB_form2.sort!)(prob,(>))
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
				sel_prob = BB_form2.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/abs(prob.GUB) >=  -TOL_GAP)
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
						@printf("                                q value= %13.6e, qmn = %13.6e, qmx = %13.6e\n",
								transpose(S[index,:])*sel_prob.x,sel_prob.lmin[index],sel_prob.lmax[index])
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
								BB_form2.insert!(prob,child[i])
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
							#println("(*) LBs = ",LBs)
							#println("(*) lb = ",lb," ub = ",ub)
							#println("(*) GLB = ",prob.GLB," GUB = ",prob.GUB," GAP = ",prob.GAP)
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
							child[i] = elem_bb_2(1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						if(prob.policy == :sort)
							(BB_form2.sort!)(prob,(>))
						end

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time
			println(prob.xUB)

			close(fid_tim)
			close(fid_GUB)
			close(fid_bab)

			return (prob.GUB,prob.GAP,prob.totnodes,prob.numlp,prob.nummilp,execution_time,0.0)
		end
	end

	########################u#############################
	# given an elem_bb_2 e, for which the lower bound
	# has already been compute, subdivide it into 2
	# subproblems using rules derived from the KKT conditions
	#####################################################
	function subdivide(prob,e::elem_bb_2,iprint)

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
		gap       = zeros(n)
		for i=1:n
			gap[i]= 0.5*(e.x[i]*transpose(Q[i,:])*e.x)-e.g[i]
		end
		maxgap_x  = gap[imaxgap_x]

		if iprint >= 1
			println("branching on ", imaxgap_x," the gap is ", maxgap_x)
		end

		if maxgap_x <= 1.e-6
			println("WARNING from subdivide: no further branching is possible!!")
			return Array{elem_bb_2,1}(undef,0)
		end
		#NOTE: branch is forcedly binary!
		nT = 2
		###################################
		# builds the 2 subproblems
		###################################
		eT = Array{elem_bb_2}(undef,nT)
		for iT = 1:nT
			eT[iT] = elem_bb_2(n)
			eT[iT].LB = e.LB
			eT[iT].UB = e.UB
			eT[iT].xUB= copy(e.xUB)
			eT[iT].x  = copy(e.x)
			eT[iT].g  = copy(e.g)
			eT[iT].xmin  = copy(e.xmin)
			eT[iT].xmax  = copy(e.xmax)
			eT[iT].lmin  = copy(e.lmin)
			eT[iT].lmax  = copy(e.lmax)
			if iT == 2
			# Q_ix>=Q_ix^*
				eT[iT].lmin[imaxgap_x] = transpose(Q[imaxgap_x,:])*eT[iT].x
				#println("lmin dopo branching",eT[iT].lmin[imaxgap_x] )
			else
				# Q_ix<=Q_ix^*
				eT[iT].lmax[imaxgap_x] = transpose(Q[imaxgap_x,:])*eT[iT].x
				#println("lmax dopo branching",eT[iT].lmin[imaxgap_x] )

			end #end if iT == 1

		end #end for iT

		return eT
	end

	function compute_lowerbound!(e::elem_bb_2,prob, tol, iprint, isroot)
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
		c             = prob.c
		T 			  = prob.T
		GUB           = prob.GUB
		low	          = e.lmin
		up            = e.lmax
		lowx          = e.xmin
		upx           = e.xmax
		wwaux         = e.x
		ttaux         = e.g
		vmap          = prob.vmap
		upperbound    = (1-tol)*(GUB-T[1])
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
			for i=1:n
				gap[i]=0.5*wwaux[i]*(Q[i,:]'*wwaux) - ttaux[i]

				#if (gap[i] > (1/n)*tol*GUB)
				if (gap[i] > 0)
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
			totlp  = solvelp
			totlp1 = solvelp
			p      = sortperm(gap,rev=true)
			ind    = p

			if BT >= 1
				#####################################
				# RISOLVO totlp + 2*totlp1 problemi
				#####################################
				low0   = copy(low)
				up0    = copy(up)
				upx0   = copy(upx)
				pmap(set_lp_2,[low0 for k in 1:nprocs()],
							[up0 for k in 1:nprocs()],
							[lowx for k in 1:nprocs()],
							[upx0 for k in 1:nprocs()],
							[upperbound for k in 1:nprocs()])
				resp   = pmap(solutore_min_l_2, [ind[k] for k in 1:totlp])
				for k = 1:totlp
					prob.numlp = prob.numlp+1
					i = ind[k]
					low[i] = resp[k][1] #mn_li
				end
				resp1   = pmap(solutore_max_l_2, [ind[k] for k in 1:totlp1])
				resp2   = pmap(solutore_max_x_2, [ind[k] for k in 1:totlp1])
				for k = 1:totlp1
					prob.numlp = prob.numlp+2
					i = ind[k]
					up[i]  = resp1[k][1] #mx_li
					upx[i] = resp2[k][1] #mx_i
					feas   = resp2[k][2]
				end#end for i=1:n
				pmap(clear_lp_2, [k for k in 1:nprocs()])
			else
				feas = true
			end
			if feas

				lp  = build_model_2(Q,c,T,A,b,Aeq,beq,low,up,lowx,upx,upperbound)

				add_Xmax_cons!(lp,n,upx)
				add_Qbound_cons_2!(lp,n,low,up,Q)
				add_MC_cons_2!(lp,n,low,up,upx,Q)

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
					e.lmin = low
					e.lmax = up
					e.xmin = lowx
					e.xmax = upx
					e.index= 1
					UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
					if iprint >= 1
						println("lowold = ",lowerboundold,"   low = ",lowerbound)
					end
					if UB < GUB
						GUB = UB
						upperb = (1-tol)*(UB-T[1])
						if iprint >= 1
							println("upperb = ",GUB)
						end
					end
					e.LB   = lowerbound

					#add_NEP_GUB_con!(lp,n,vmap,upperb,minimo)

					if UB < e.UB
						e.UB  = UB
						e.xUB = copy(x)
					end
					if (UB < prob.GUB)
						prob.GUB = UB
						prob.xUB = x
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
				@printf(" solvelp = %3d totlp = %3d totlp1 = %3d lowerbound: %15.8e\n",solvelp,totlp,totlp1,lowerbound);
				@printf(" lowerbound - lowerboundold = %15.8e max(0.5*tol,0.01*(GUB-lowerbound)) = %15.8e\n",lowerbound - lowerboundold,max(0.5*tol,0.01*(GUB-lowerbound)))
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
			for i=1:n
				gap[i]=0.5*wwaux[i]*(Q[i,:]'*wwaux) - ttaux[i]
			end
			p      = sortperm(gap,rev=true)
			e.index= p[1]
			#println("branch index: ",p[1])
		end
		return (lowerbound, e.UB, solvetime)

	end

	function add_Xmax_cons!(lp,n,upx)
		Gurobi.set_dblattrarray!(lp,"UB",1,n,upx)
		Gurobi.update_model!(lp)
	end

	function compute_strong_McCormick_bound!(e::elem_bb_2,prob,tol,low,up,lowx,upx,iprint)
		n       = prob.n
		m       = prob.m
		meq		= prob.meq
		A       = prob.A
		Aeq     = prob.Aeq
		b       = prob.b
		beq     = prob.beq
		Q       = prob.Q
		c   	= prob.c
		T 		= prob.T
		GUB     = prob.GUB
		vmap    = prob.vmap
		lowold  = -Inf
		lowerb  = 0.0
		upperb  = (1.0-tol)*(GUB-T[1])
		feas    = true
		ww      = zeros(n)
		tt      = zeros(n)
		solvelp = 0.2*n

		start_time = time()

		#while ((lowerb - lowold) > max(tol,0.1*tol*GUB))
		while true
			if iprint >= 1
				@printf("lowerb-lowold = %15.8e -- max(tol,0.1*tol*GUB) = %15.8e\n",(lowerb - lowold),max(tol,0.1*tol*GUB))
			end

			lp  = build_model_2(Q,c,T,A,b,Aeq,beq,low,up,lowx,upx,upperb)

			if (lowold < 0)
				gap     = ones(n)
				totlp   = n
			else
				gap     = 0.5*(ww'*(Q*ww)).-tt
				totlp   = solvelp
			end
			num_LP = convert(Int,totlp)
			p      = sortperm(gap,rev=true)
			ind    = p[1:num_LP]

			###############################
			# RISOLVO 3*num_LP problemi
			###############################
			for k = 1:num_LP
				i = ind[k]
				(mn_li, feas) = compute_min_l(lp,prob,i)
				(mx_li, feas) = compute_max_l(lp,prob,i)
				(mx_i,  feas) = compute_max_x(lp,prob,i)
				low[i] = mn_li
				up[i]  = mx_li
				upx[i] = mx_i
				if iprint >= 1
					@printf(" low[%3d] = %13.6e up[%3d] = %13.6e upx[%3d] = %13.6e\n",i,low[i],i,up[i],i,upx[i])
				end
			end#end for i=1:n

			if (feas)
				add_MC_cons_2!(lp,n,low,up,upx,Q)
				add_GUB_con_2!(lp,n,c,T[1],upperb)
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
					e.lmin = low
					e.lmax = up
					e.xmin = lowx
					e.xmax = upx
					e.index= 1
					if iprint >= 1
						println("lowold = ",lowold,"   low = ",lowerb)
					end
					UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
					if UB < GUB
						GUB = UB
						upperb = (1-tol)*(UB-T[1])
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

	function compute_low(lp,prob)
		#lp = Gurobi.copy(lpin)
		Gurobi.set_sense!(lp,:minimize)
		coefs = zeros(prob.nvar)
		coefs[prob.vmap["x"]] = prob.c
		coefs[prob.vmap["g"]] = ones(prob.n)
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
		addOption(IPprob,"print_level",1)
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
			println("solution is ",fstar,iter)
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
