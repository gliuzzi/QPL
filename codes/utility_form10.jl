__precompile__()

#!============================================================================================
#!    QPL - A computational study on QP problems with general linear constraints
#!    Copyright (C) 2021  G.Liuzzi, M.Locatelli, V.Piccialli
#!
#!    This program is free software: you can redistribute it and/or modify
#!    it under the terms of the GNU General Public License as published by
#!    the Free Software Foundation, either version 3 of the License, or
#!    (at your option) any later version.
#!
#!    This program is distributed in the hope that it will be useful,
#!    but WITHOUT ANY WARRANTY; without even the implied warranty of
#!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#!    GNU General Public License for more details.
#!
#!    You should have received a copy of the GNU General Public License
#!    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#!
#!    G. Liuzzi, M. Locatelli, V. Piccialli. A computational study on QP problems
#!    with general linearconstraints. Submitted to Optimization Letters (2021)
#!
#!============================================================================================

module utility_form10
	using Distributed
	using BB_form10
	@everywhere using partools_form10
	using JuMP
	using CPLEX
	using Ipopt
	using MAT
	using LinearAlgebra
	using Printf
	using Random

	export compute_rootnode_gap_10
	export solve_10
	export solve_10_old

	function compute_rootnode_gap_10(DATA,
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
		# 	open problems are selected according to best bound
		#	compute LB after branching
		#####################################################

		if(MAXNODESin <= 0)
			println("WARNING: passed value for MAXNODES is <= 0, B&B will explore 0 nodes!")
			MAXNODES = 0
		else
			MAXNODES = MAXNODESin
		end
		if(TOLGAP_IN > 0)
			TOL_GAP = TOLGAP_IN
		end

		prob         = BB_form10.BB_10()
		LBs          = Array{Float64}(undef,0)

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
			error = true
			return 0.0, Inf, prob, LBs, solved, error
		end
		(n,h1)   = size(Q)
		(m,h1)   = size(A)
		(meq,h1) = size(Aeq)

		Qpos = DATA["Qpos"]
		Dn   = DATA["Dn"]
		Un   = DATA["Un"]

		prob.n   = n
		prob.m   = m
		prob.meq = meq
		prob.Q   = copy(Q)
		prob.Qpos= copy(Qpos)
		prob.Dn  = copy(Dn)
		prob.Un  = copy(Un)
		prob.c   = copy(c)
		prob.T   = copy(T)
		prob.A   = copy(A)
		prob.Aeq = copy(Aeq)
		prob.b   = copy(b)
		prob.beq = copy(beq)
		################ INPUT DATA  END   ##################

		p = length(Dn)
		if (p == 0)
			println("NOTICE: the problem is convex, no need for B&B!!!")
			start_time = time()
			(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,1)
			prob.GUB = bestval
			prob.xUB = copy(xstar)
			prob.GAP = 0.0
			execution_time = time()-start_time

			solved = true
			error  = false
			return execution_time, prob.GAP, prob, LBs, solved, error

			return (prob.GUB,prob.GAP,1,0,0,execution_time,0.0)
		end
		prob.nvar  = convert(Int,n + 2*p)
		prob.vmap  = Dict("x" => 1:n,
						  "z" => n+1:n+p,
						  "g" => n+p+1:n+p+p)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_10, [n for i in 1:nprocs()],
						   [p for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [meq for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
						   [Qpos for i in 1:nprocs()],
						   [Dn for i in 1:nprocs()],
						   [Un for i in 1:nprocs()],
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
			GAP_time = 0.0
			GAP_node = 0
			if (nloc >0)
				(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,nloc)
				#xstar = [0.25838044394163306, 0.0, 0.0, 0.19120247537184457, 0.0, 0.3433121750327294, 0.0, 0.0, 0.6488588197152324, 0.0, 0.0, 0.0, 0.0, 0.573212917186699, 0.0, 0.0, 0.06177391227639872, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02199876127776128, 0.07204154466034857, 0.1694256448634834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37803804931411833, 0.0, 0.7416802440298111, 0.0, 0.0, 0.9107640307578458, 0.0, 0.0, 0.0, 0.0, 0.23640045712434524, 0.0, 0.0, 0.0, 0.0]
				#bestval = transpose(xstar)*Q*xstar/2 + (transpose(c)*xstar)[1] + T[1]
				prob.GUB = bestval
				prob.xUB = copy(reshape(xstar,(n,)))
				println(prob.xUB)
				#readline()
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
			lowz          = zeros(p)
			upz           = ones(p)
			for i in 1:p
				lowz[i]      = sum(Un[findall(x -> x<0,Un[:,i]),i])
				upz[i]       = sum(Un[findall(x -> x>0,Un[:,i]),i])
			end
			#ww            = zeros(n)
			#tt            = zeros(n)
			println("build initial model (w/o Qbounds and MC constrs)")
			lp = CPXbuild_initial_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,LB,UB,lowz,upz,upperboundmod)

			#readline()

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = CPXfind_feasible_point(lp,prob)
				#print(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				#println(size(Un')," ",size(diagm(Dn))," ",size(Un))
				println("CONTROLLO obj: ",prob.GUB," decomposed: ",(c'*x + x'*Un*diagm(Dn)*Un'*x/2 + x'*Qpos*x/2 + T)[1])
				upperbound = (1-tol)*prob.GUB
				upperboundmod = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(reshape(xstar,(n,)))
				println(prob.xUB)
				#lp = CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,LB,UB,lowz,upz,upperboundmod)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end
			lp = 0

			e = elem_bb_10(n,p)
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,lowx,upx,lowz,upz,iprint)
			#println("    LB = ",lb,"     GUB = ",prob.GUB," e.LB = ",e.LB)
			#println("     x = ",e.xUB)
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

			(BB_form10.sort!)(prob,(>))

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

			pmap(garbage_collection, [k for k in 1:nprocs()])
			GC.gc()

			solved = false
			error  = false
			return rootgap_time, prob.GAP, prob, LBs, solved, error
		end
	end

	function solve_10(DATA,prob,LBs,
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
		(n,h1)   = size(Q)
		(m,h1)   = size(A)
		(meq,h1) = size(Aeq)

		Qpos = DATA["Qpos"]
		Dn   = DATA["Dn"]
		Un   = DATA["Un"]
		################ INPUT DATA  END   ##################

		p = length(Dn)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_10, [n for i in 1:nprocs()],
						   [p for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [meq for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
						   [Qpos for i in 1:nprocs()],
						   [Dn for i in 1:nprocs()],
						   [Un for i in 1:nprocs()],
						   [c for i in 1:nprocs()],
						   [T for i in 1:nprocs()],
						   [A for i in 1:nprocs()],
						   [Aeq for i in 1:nprocs()],
						   [b for i in 1:nprocs()],
						   [beq for i in 1:nprocs()],
						   [prob.vmap for i in 1:nprocs()])
		######## SET DATA FOR PARALLEL PROCESSING END #######

		if (p == 0)
			println("NOTICE: the problem is convex, no need for B&B!!!")
			start_time = time()
			(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,1)
			prob.GUB = bestval
			prob.xUB = copy(xstar)
			prob.GAP = 0.0
			execution_time = time()-start_time
			return (prob.GUB,prob.GAP,1,0,0,execution_time,0.0)
			#return
		end


		fid_GUB = open("GUB_stat.txt","w")
		fid_tim = open("tim_stat.txt","w")
		fid_bab = open("bab_stat.txt","w")

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
			lowz          = zeros(p)
			upz           = ones(p)
			for i in 1:p
				lowz[i]      = sum(Un[findall(x -> x<0,Un[:,i]),i])
				upz[i]       = sum(Un[findall(x -> x>0,Un[:,i]),i])
			end

			while (prob.nnodes > 0) && (prob.totnodes < MAXNODES) && (time()-start_time <= time_limit)
				sel_prob = BB_form10.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/max(1.00,abs(prob.GUB)) >=  -TOL_GAP)
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
									prob.xUB = child[i].xUB
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
									@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
									flush(fid_GUB)
								end
							end

							if iprint >= 1
								if (child[i].LB > child[i].UB + TOL_GAP)
									println("\nWarning: UB lower than LB!!!")
									@printf(fid_bab,"\nWarning: UB lower than LB!!!")
									#println("a.LB = ",child[i].LB," a.UB = ",child[i].UB," GUB = ",prob.GUB)
									#println("a.xUB =", a.xUB)
									@printf(fid_bab,"a.LB = %7.4f, a.UB = %7.4f, GUB = %7.4f\n",child[i].LB,child[i].UB,prob.GUB)
									flush(fid_bab)
								end
							end

							if child[i].LB < prob.GUB*(1 - TOL_GAP)
								BB_form10.insert!(prob,child[i])
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
							child[i] = elem_bb_10(1,1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						(BB_form10.sort!)(prob,(>))

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time
			x = prob.xUB
			println(x)
			println("funzione obiettivo: ",0.5*(x'*Q*x) + (c'*x)[1] + T[1])
			println("veq: ",Aeq*x - beq)
			println("vin: ",A*x-b)
			println("x: ",x)
			pmap(garbage_collection, [k for k in 1:nprocs()])
			GC.gc()
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

	function solve_10_old(DATA,
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
		# 	open problems are selected according to best bound
		#	compute LB after branching
		#####################################################

		if(MAXNODESin <= 0)
			println("WARNING: passed value for MAXNODES is <= 0, B&B will explore 0 nodes!")
			MAXNODES = 0
		else
			MAXNODES = MAXNODESin
		end
		if(TOLGAP_IN > 0)
			TOL_GAP = TOLGAP_IN
		end

		prob         = BB_form10.BB_10()
		LBs          = Array{Float64}(undef,0)

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

		#LB,UB,c,Q,A,b,Aeq,beq,T,flag = values(DATA)

		if !(flag)
			println("Some upper or lower bound on the variables is set to infinity. Exiting \n\n")
			return
		end
		(n,h1)   = size(Q)
		(m,h1)   = size(A)
		(meq,h1) = size(Aeq)

		Qpos = DATA["Qpos"]
		Dn   = DATA["Dn"]
		Un   = DATA["Un"]

		#Qpos, Dn, Un = decompose_Q(Q)

		prob.n   = n
		prob.m   = m
		prob.meq = meq
		prob.Q   = copy(Q)
		prob.Qpos= copy(Qpos)
		prob.Dn  = copy(Dn)
		prob.Un  = copy(Un)
		prob.c   = copy(c)
		prob.T   = copy(T)
		prob.A   = copy(A)
		prob.Aeq = copy(Aeq)
		prob.b   = copy(b)
		prob.beq = copy(beq)
		################ INPUT DATA  END   ##################

		p = length(Dn)
		if (p == 0)
			println("NOTICE: the problem is convex, no need for B&B!!!")
			start_time = time()
			(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,1)
			prob.GUB = bestval
			prob.xUB = copy(xstar)
			prob.GAP = 0.0
			execution_time = time()-start_time
			return (prob.GUB,prob.GAP,1,0,0,execution_time,0.0)
			#return
		end
		prob.nvar  = convert(Int,n + 2*p)
		prob.vmap  = Dict("x" => 1:n,
						  "z" => n+1:n+p,
						  "g" => n+p+1:n+p+p)

		######## SET DATA FOR PARALLEL PROCESSING ###########
		pmap(set_matrices_10, [n for i in 1:nprocs()],
						   [p for i in 1:nprocs()],
						   [m for i in 1:nprocs()],
						   [meq for i in 1:nprocs()],
						   [prob.nvar for i in 1:nprocs()],
						   [Q for i in 1:nprocs()],
						   [Qpos for i in 1:nprocs()],
						   [Dn for i in 1:nprocs()],
						   [Un for i in 1:nprocs()],
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
			GAP_time = 0.0
			GAP_node = 0
			if (nloc >0)
				(bestval,xstar) = find_initial_GUB(Q,c,T,A,b,Aeq,beq,nloc)
				#xstar = [0.25838044394163306, 0.0, 0.0, 0.19120247537184457, 0.0, 0.3433121750327294, 0.0, 0.0, 0.6488588197152324, 0.0, 0.0, 0.0, 0.0, 0.573212917186699, 0.0, 0.0, 0.06177391227639872, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02199876127776128, 0.07204154466034857, 0.1694256448634834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37803804931411833, 0.0, 0.7416802440298111, 0.0, 0.0, 0.9107640307578458, 0.0, 0.0, 0.0, 0.0, 0.23640045712434524, 0.0, 0.0, 0.0, 0.0]
				#bestval = transpose(xstar)*Q*xstar/2 + (transpose(c)*xstar)[1] + T[1]
				prob.GUB = bestval
				prob.xUB = copy(reshape(xstar,(n,)))
				println(prob.xUB)
				#readline()
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
			lowz          = zeros(p)
			upz           = ones(p)
			for i in 1:p
				lowz[i]      = sum(Un[findall(x -> x<0,Un[:,i]),i])
				upz[i]       = sum(Un[findall(x -> x>0,Un[:,i]),i])
			end
			#ww            = zeros(n)
			#tt            = zeros(n)
			println("build initial model (w/o Qbounds and MC constrs)")
			lp = CPXbuild_initial_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,LB,UB,lowz,upz,upperboundmod)

			#readline()

			if (nloc <= 0)
				#find one feasible point and use that as GUB
				(xstar,feas) = CPXfind_feasible_point(lp,prob)
				#print(lp)
				x = xstar
				prob.GUB = (x'*Q*x/2 + c'*x + T)[1]
				#println(size(Un')," ",size(diagm(Dn))," ",size(Un))
				println("CONTROLLO obj: ",prob.GUB," decomposed: ",(c'*x + x'*Un*diagm(Dn)*Un'*x/2 + x'*Qpos*x/2 + T)[1])
				upperbound = (1-tol)*prob.GUB
				upperboundmod = (1-tol)*(prob.GUB-T[1])
				prob.xUB = copy(reshape(xstar,(n,)))
				println(prob.xUB)
				#lp = CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,LB,UB,lowz,upz,upperboundmod)
				println(" upperbound: ",upperbound)
				println("feasibility: ",feas)
			end
			lp = 0

			e = elem_bb_10(n,p)
			e.UB = prob.GUB
			e.xUB= prob.xUB
			(lb, ub, solvetime) = compute_strong_McCormick_bound!(e,prob,tol,lowx,upx,lowz,upz,iprint)
			println("    LB = ",lb,"     GUB = ",prob.GUB," e.LB = ",e.LB)
			println("     x = ",e.xUB)
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

			(BB_form10.sort!)(prob,(>))

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
				sel_prob = BB_form10.extract!(prob)
				if iprint >= 1
					println("sel_prob.LB & UB = ",sel_prob.LB," ",sel_prob.UB," GUB = ",prob.GUB)
				end
				(v,i) = findmin(abs.(LBs.-sel_prob.LB))
				deleteat!(LBs,i)
				if ((sel_prob.LB-prob.GUB)/max(1.00,abs(prob.GUB)) >=  -TOL_GAP)
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
									prob.xUB = child[i].xUB
									prob.GAP = (prob.GUB - prob.GLB)/max(1.0,abs(prob.GUB))
									@printf(fid_GUB,"GUB = %20f after %20d B&B nodes\n",prob.GUB,prob.totnodes)
									flush(fid_GUB)
								end
							end

							if (child[i].LB > child[i].UB + TOL_GAP)
								println("\nWarning: UB lower than LB!!!")
								@printf(fid_bab,"\nWarning: UB lower than LB!!!")
								#println("a.LB = ",child[i].LB," a.UB = ",child[i].UB," GUB = ",prob.GUB)
								#println("a.xUB =", a.xUB)
								@printf(fid_bab,"a.LB = %7.4f, a.UB = %7.4f, GUB = %7.4f\n",child[i].LB,child[i].UB,prob.GUB)
								flush(fid_bab)
							end

							if child[i].LB < prob.GUB*(1 - TOL_GAP)
								BB_form10.insert!(prob,child[i])
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
							child[i] = elem_bb_10(1,1)
						end
						flush(fid_tim)
						flush(fid_GUB)
						flush(fid_bab)

						(BB_form10.sort!)(prob,(>))

						child = 0
						#gc()
					end #if !(child == null)
				end # begin
			end

			## QUESTO return è solo per DEBUG, poi va tolto!
			execution_time = time()-start_time

			######## SET DATA FOR PARALLEL PROCESSING ###########
			pmap(clear_matrices_10, [i for i in 1:nprocs()])
			######## SET DATA FOR PARALLEL PROCESSING END #######

			x = prob.xUB
			println(x)
			println("funzione obiettivo: ",0.5*(x'*Q*x) + (c'*x)[1] + T[1])
			println("veq: ",Aeq*x - beq)
			println("vin: ",A*x-b)
			println("x: ",x)
			return (prob.GUB,prob.GAP,prob.totnodes,prob.numlp,prob.nummilp,execution_time,0.0)
		end
	end

	########################u#############################
	# given an elem_bb_10 e, for which the lower bound
	# has already been compute, subdivide it into 2
	# subproblems using rules derived from the KKT conditions
	#####################################################
	function subdivide(prob,e::elem_bb_10,iprint)

		n             = prob.n
		m             = prob.m
		p             = length(prob.Dn)
		meq           = prob.meq
		A             = prob.A
		Aeq           = prob.Aeq
		b             = prob.b
		beq           = prob.beq
		Q             = prob.Q
		c             = prob.c
		T             = prob.T
		vmap	      = prob.vmap

		imaxgap_x = e.index
		#println(e.index)

		gap       = zeros(p)
		for j = 1:p
			gap[j] = e.g[j] - e.z[j]^2
		end
		maxgap_x  = gap[imaxgap_x]

		if iprint >= 1
			println("branching on ", imaxgap_x," the gap is ", maxgap_x)
			#println("gap: ",gap)
		end

		if maxgap_x <= 1.e-6
			println("WARNING from subdivide: no further branching is possible!!")
			println("e.LB:",e.LB," GUB:",prob.GUB," GAP:",(prob.GUB - e.LB)/max(1.0,abs(prob.GUB)))
			println("g^TDn/2 = ",(transpose(prob.Dn)*e.g)/2," term = ",transpose(prob.Dn)*(e.z.^2)/2)
			return Array{elem_bb_10,1}(undef,0)
		end

		nT = 2
		###################################
		# builds the 2 subproblems
		###################################
		eT = Array{elem_bb_10}(undef,nT)
		for iT = 1:nT
			eT[iT] = elem_bb_10(n,p)
			eT[iT].LB = e.LB
			eT[iT].UB = e.UB
			eT[iT].xUB= copy(e.xUB)
			eT[iT].x  = copy(e.x)
			eT[iT].z  = copy(e.z)
			eT[iT].g  = copy(e.g)
			eT[iT].xmin  = copy(e.xmin)
			eT[iT].xmax  = copy(e.xmax)
			eT[iT].zmin = copy(e.zmin)
			eT[iT].zmax = copy(e.zmax)
			eT[iT].index = e.index
			if iT == 2
				eT[iT].zmin[imaxgap_x] = e.z[imaxgap_x]
			else
				eT[iT].zmax[imaxgap_x] = e.z[imaxgap_x]
			end #end if iT == 1

		end #end for iT

		return eT
	end

	function compute_lowerbound!(e::elem_bb_10,prob, tol, iprint, isroot)
		#questo bound è usato solo ai nodi INTERMEDI
		#######################################
		# BT = 0  NESSUN BOUND TIGHTENING
		# BT = 1  1 GIRO DI BOUND TIGHTENING
		# BT = Inf  STRONG BOUND TIGHTENING
		#######################################
		BT            = Inf

		n             = prob.n
		m             = prob.m
		p             = length(prob.Dn)
		meq           = prob.meq
		A             = prob.A
		Aeq           = prob.Aeq
		b             = prob.b
		beq           = prob.beq
		Q             = prob.Q
		Qpos          = prob.Qpos
		Dn            = prob.Dn
		Un            = prob.Un
		c             = prob.c
		T             = prob.T
		GUB           = prob.GUB
		lowx          = e.xmin
		upx           = e.xmax
		lowz          = e.zmin
		upz           = e.zmax
		ww            = e.x
		tt            = e.z
		ff            = e.g
		vmap          = prob.vmap
		upperbound    = (1-tol)*GUB
		upperboundmod = (1-tol)*(GUB-T[1])
		feas          = true
		num_lb        = 0
		solvelp       = 0.2*n
		lowerbound    = e.LB #0
		originallower = lowerbound
		lowerboundold = originallower-1
		lowsafe       = originallower

		#lp  = CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)

		start_time =time()
		gap = zeros(p)

		if iprint >= 1
			println("================== compute_lowerbound! ================================")
		end

		#for ibt in [1 ]
		while true
			solvelp = 0
			for j = 1:p
				gap[j] = ff[j] - tt[j]^2
				if gap[j] > 0
					solvelp += 1
				end
			end

			totlp1 = solvelp
			pord   = sortperm(gap,rev=true)
			ind    = pord

			if BT >= 1

				###########################################
				# RISOLVO i problemi per bound tightening
				###########################################
				lowz0 = copy(lowz)
				upz0  = copy(upz)
				pmap(set_lp_10,[lowx for k in 1:nprocs()],
							[upx for k in 1:nprocs()],
							[lowz0 for k in 1:nprocs()],
							[upz0 for k in 1:nprocs()],
							[upperboundmod for k in 1:nprocs()])

				if iprint >= 2
					for k = 1:p
						@printf("(0): gap[%3d] = %13.6e\n",k,gap[ind[k]])
					end
				end
				if iprint >= 2
					for k = 1:p
						@printf("(1): lowz[%3d] = %13.6e upz[%3d] = %13.6e\n",k,lowz[k],k,upz[k])
					end
				end
				#readline()

				resp1   = pmap(CPXsolutore_max_z_10, [ind[k] for k in 1:totlp1])
				resp2   = pmap(CPXsolutore_min_z_10, [ind[k] for k in 1:totlp1])
				feas = true
				for k = 1:totlp1
					prob.numlp = prob.numlp+2
					i = ind[k]
					flag   = resp2[k][2]
					if flag == 1
						lowz[i] = resp2[k][1] #mn_i
					else #if flag == 2
						feas = false
					end

					flag   = resp1[k][2]
					if flag == 1
						upz[i] = resp1[k][1] #mn_i
					else #if flag == 2
						feas = false
					end
					if iprint >= 1
						@printf("(2): lowz[%3d] = %13.6e upz[%3d] = %13.6e\n",i,lowz[i],i,upz[i])
					end
				end#end for i=1:n
				pmap(clear_lp_10, [k for k in 1:nprocs()])
			else
				feas = true
			end

			#readline()
			e.zmin = copy(lowz)
			e.zmax = copy(upz)
			if feas
				#lp = 0
				try
					empty!(lp)
				catch
				end
				lp = CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)

				############################################
				# CALCOLO il LOWERBOUND con CPLEX
				############################################
				(lowerbound1,sol,flag) = CPXcompute_low(lp,prob)
				if iprint >= 1
					println("***+++***+++***+++***+++ ",termination_status(lp)," ",primal_status(lp)," ",dual_status(lp)," ",flag)
					println("low1 = ",lowerbound1," lowold = ",lowerboundold)
				end
				lp = 0
				#####################################################################
				# flag = 1 : CPLEX OPTIMAL (ma dobbiamo controllare la feasibility)
				# flag = 2 : CPLEX INFEASIBLE
				# flag = 3 : CPLEX NUMERICAL ERROR (try IPOPT)
				#####################################################################
				if flag == 1
					if iprint >= 1
						println("CPLEX says it solved the problem")
					end
					############# CPLEX says it solved the problem, but we hav to check feasibility
					x      = reshape(sol[vmap["x"]],(n,))
					z      = reshape(sol[vmap["z"]],(p,))
					g      = reshape(sol[vmap["g"]],(p,))
					flag_error_ipopt = false
					if check_feasibility(x,z,g,tol,Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod) == 0
						if iprint >= 1
							println("OK, CPLEX says it solved the problem, but the solution is INFEASIBLE")
						end
						# OK, CPLEX says it solved the problem, but the solution is INFEASIBLE
						# so we solve the problem again with IPOPT
						try
							empty!(ipopt_prob)
						catch
						end
						ipopt_prob = IPObuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod,tol)
						(lowerbound1,sol,flag) = IPOcompute_low(ipopt_prob,prob)
						if flag == 3
							if iprint >= 1
								println("IPOPT report NUMERICAL ERROR")
							end
							lowerbound = lowsafe
							flag_error_ipopt = true
						else
							if iprint >= 1
								println("IPOPT solved the problem")
								#println("CONTR: ",sol," ",flag)
							end
							x      = reshape(sol[vmap["x"]],(n,))
							z      = reshape(sol[vmap["z"]],(p,))
							g      = reshape(sol[vmap["g"]],(p,))
							if check_feasibility(x,z,g,tol,Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod) == 0
								if iprint >= 1
									println("OK, IPOPT says it solved the problem, but the solution is INFEASIBLE")
								end
								flag_error_ipopt = true
								lowerbound = lowsafe
							else
								if iprint >= 1
									println("OK, IPOPT solved the problem")
								end
								lowerbound = lowerbound1
								lowsafe= lowerbound
							end
						end
						ipopt_prob = 0
					else
						if iprint >= 1
							println("CPLEX says it solved the problem and solution is feasible")
						end
						lowerbound = lowerbound1
						lowsafe= lowerbound
					end
					if ~flag_error_ipopt
						ww     = x
						tt     = z
						ff     = g
						e.LB   = lowerbound
						e.x    = copy(x)
						e.z    = copy(z)
						e.g    = copy(g)
						e.zmin = lowz
						e.zmax = upz
						#e.index= 1
						if flag_error_ipopt
							UB = -1000000.0
						else
							UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
						end
						if iprint >= 1
							println("lowold = ",lowerboundold,"   low = ",lowerbound," UB = ",UB)
						end

						if UB < GUB
							GUB = UB
							xUB = reshape(x,(n,))
							upperb = (1-tol)*UB
							upperboundmod = (1-tol)*(GUB-T[1])
						end

						if UB < e.UB
							e.UB  = UB
							e.xUB = copy(x)
							e.x = copy(x)
						end
					end
				elseif (flag == 2)
					if iprint >= 1
						println("CPLEX reported not feas (1)")
					end
					break
				else # flag = 3
					if iprint >= 1
						# CPLEX says numerical error or other issue, so we try IPOPT
						println("flag = 3, trying ipopt")
					end



					flag_error_ipopt = false
					lowerbound = lowsafe
					try
						empty!(ipopt_prob)
					catch
					end
					ipopt_prob = IPObuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod,tol)
					(lowerbound1,sol,flag) = IPOcompute_low(ipopt_prob,prob)
					if flag == 3
						if iprint >= 1
							println("IPOPT reports NUMERICAL ERROR")
						end
						lowerbound = lowsafe
						flag_error_ipopt = true
					else
						#println("CONTR: ",sol," ",flag)
						x      = reshape(sol[vmap["x"]],(n,))
						z      = reshape(sol[vmap["z"]],(p,))
						g      = reshape(sol[vmap["g"]],(p,))
						if check_feasibility(x,z,g,tol,Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod) == 0
							if iprint >= 1
								println("IPOPT solution is infeasible")
								#println("Violatooooooooo  e DUEEEEEEE")
							end
							flag_error_ipopt = true
							lowerbound = lowsafe
						else
							if iprint >= 1
								println("IPOPT solution is OK")
							end
							lowerbound = lowerbound1
							lowsafe= lowerbound
						end
					end
					ipopt_prob = 0
					if ~flag_error_ipopt
						ww     = x
						tt     = z
						ff     = g
						e.LB   = lowerbound
						e.x    = copy(x)
						e.z    = copy(z)
						e.g    = copy(g)
						e.zmin = lowz
						e.zmax = upz
						#e.index= 1
						UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]

						if iprint >= 1
							println("lowold = ",lowerboundold,"   low = ",lowerbound," UB = ",UB)
						end

						if UB < GUB
							GUB = UB
							xUB = reshape(x,(n,))
							upperb = (1-tol)*UB
							upperboundmod = (1-tol)*(GUB-T[1])
						end

						if UB < e.UB
							e.UB  = UB
							e.xUB = copy(x)
							e.x = copy(x)
						end
					end


					if (flag == 2)
						if iprint >= 1
							println("IPOPT reported problem is not feasible")
						end
						# IPOPT reported problem NOT FEASIBLE, break (ma questo è difficile che accada)
						break
					end
				end
			else
				if iprint >= 1
					println("BOUND TIGHTENING reported not feas (2)")
				end
				break
			end
			if iprint >= 1
				#@printf(" solvelp = %3d lowerbound: %15.8e\n",solvelp,lowerbound);
				@printf("lowerb-lowold = %15.8e -- max(0.5*tol,0.01*(GUB-lowerb)) = %15.8e\n",(lowerbound - lowerboundold),max(0.5*tol,0.01*(GUB-lowerbound)))
			end

			# criterio di arresto per il BT
			if ((lowerbound - lowerboundold) <= max(0.5*tol,0.01*(GUB-lowerbound)))
				break
			end
			if isroot
				break
			end
			if BT <= 1
				break
			end
			lowerboundold = lowerbound
		end #while

		solvetime = time()-start_time
		#println("computed LB=",lowerbound)
		if ~feas
			flag = 2
		end
		if (flag == 2)
			# CPLEX, IPOPT or BoundTightenin reported INFEASIBILITY
			#println("LB not feasible, becareful the UB could have been updated")
			e.LB = Inf
			#e.UB = Inf
			lowerbound = Inf
		elseif flag == 1
			# we successfully computed an LB, we update the branching index e return
			gap       = zeros(p)
			for j = 1:p
				gap[j] = ff[j] - tt[j]^2
			end
			pord      = sortperm(gap,rev=true)
			e.index= pord[1]
		else
			# there was an error (flag = 3) so the LB stays at the father value
			# we update the branching index and return
			gap       = zeros(p)
			for j = 1:p
				gap[j] = ff[j] - tt[j]^2
			end
			pord      = sortperm(gap,rev=true)
			#if e.index == pord[1]
			#	e.index = pord[2]
			#else
			#	e.index= pord[1]
			#end
			e.index += 1
			if e.index > p
				e.index = 1
			end
			if iprint >= 1
				if (lowsafe >= originallower)
					println("WARNING: lower bound does not improve")
					println("lowsafe = ",lowsafe)
					println("origlow = ",originallower)
					println("lowerbn = ",lowerbound)
				end
			end
		end
		return (lowerbound, e.UB, solvetime)

	end

	function compute_strong_McCormick_bound!(e::elem_bb_10,prob,tol,lowx,upx,lowz,upz,iprint)
		n       = prob.n
		p       = length(lowz)
		m       = prob.m
		meq		= prob.meq
		A       = prob.A
		Aeq     = prob.Aeq
		b       = prob.b
		beq     = prob.beq
		Q       = prob.Q
		Qpos    = prob.Qpos
		Dn      = prob.Dn
		Un      = prob.Un
		c   	= prob.c
		T 		= prob.T
		GUB     = prob.GUB
		xUB     = prob.xUB
		vmap    = prob.vmap
		lowold  = -Inf
		lowerb  = -Inf #0.0
		lowsafe = e.LB
		upperb  = (1.0-tol)*GUB
		upperbmod  = (1.0-tol)*(GUB-T[1])
		feas    = true
		ww      = zeros(n)
		tt      = zeros(p)
		ff      = zeros(p)
		solvelp = 0.2*n

		count_while = 1
		start_time = time()

	#	while ((lowerb - lowold) > max(tol,0.1*tol*GUB))
		while true
			try
				empty!(lp)
			catch
			end
			lp  = 0
			lp  = CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperbmod)

			gap     = ones(p)
			if (lowold < 0)
				gap     = ones(p)
				totlp   = p
			else
				totlp = 0
				for j = 1:p
					gap[j] = ff[j] - tt[j]^2
					if gap[j] > 0
						totlp += 1
					end
				end
			end

			num_LP = convert(Int,totlp)
			pord   = sortperm(gap,rev=true)
			ind    = pord[1:num_LP]
			#println(gap)
			#println(ind)
			#println(num_LP)

			###############################
			# RISOLVO 3*num_LP problemi
			###############################
			feas = true
			for k = 1:num_LP
				i = ind[k]
				(mn_i,  flag) = CPXcompute_min_z(lp,prob,i)
				if flag == 1
					lowz[i]= mn_i
				end
				if flag == 2
					feas = false
				end
				(mx_i,  flag) = CPXcompute_max_z(lp,prob,i)
				if flag == 1
					upz[i] = mx_i
				end
				if flag == 2
					feas = false
				end
				if iprint >= 1
					@printf("(3): lowz[%3d] = %13.6e upz[%3d] = %13.6e %s\n",i,lowz[i],i,upz[i],feas)
				end
			end#end for i=1:n
			#readline()
			e.zmin = lowz
			e.zmax = upz

			if (feas)
				#add_MC_cons!(lp,n,p,lowz,upz)
				#for i in 1:p
				#	JuMP.delete(lp,lp[:MC][i])
				#end
				#delete!(lp.obj_dict,:MC)
				#@constraint(lp, MC[i=1:p], -lp[:z][i]*(lowz[i]+upz[i]) + lp[:g][i] <= -lowz[i]*upz[i])
				for i in 1:p
					@constraint(lp, -lp[:z][i]*(lowz[i]+upz[i]) + lp[:g][i] <= -lowz[i]*upz[i])
				end

				for i in 1:p
					set_upper_bound(lp[:z][i], upz[i])
					set_lower_bound(lp[:z][i],lowz[i])
				end

				lowold  = lowerb
				lowsafe = lowerb
				############################################
				# CALCOLO il LOWERBOUND
				############################################
				#println("avvio calcolo LB:")
				(lowerb,sol,flag) = CPXcompute_low(lp,prob)

				#println("FIN QUI ",feas," ",T[1]," ",p)
				if iprint >= 2
					if (count_while < 10)
						#write_model(lp,string("test00",string(count_while),".lp"))
					elseif (count_while < 100)
						#write_model(lp,string("test0",string(count_while),".lp"))
					else
						#write_model(lp,string("test",string(count_while),".lp"))
					end
				end
				count_while += 1

				#readline()
				if flag == 1
					if iprint >= 1
						@printf("lowerb-lowold = %15.8e -- max(tol,0.1*tol*abs(GUB)) = %15.8e -- lowerb = %15.8e\n",(lowerb - lowold),max(tol,0.1*tol*abs(GUB)),lowerb)
						#println("CONTROLLO obj: ",upperbmod," decomposed: ",(1.0-tol)*(c'*xUB + xUB'*Un*diagm(Dn)*Un'*xUB/2 + xUB'*Qpos*xUB/2))
					end
					x      = sol[vmap["x"]]
					z      = sol[vmap["z"]]
					g      = sol[vmap["g"]]
					ww     = x
					tt     = z
					ff     = g
					e.LB   = lowerb
					e.x    = copy(x)
					e.z    = copy(z)
					e.g    = copy(g)
					e.xmin = lowx
					e.xmax = upx
					e.zmin = lowz
					e.zmax = upz
					e.index= 1
					UB     = 0.5*(x'*Q*x) + (c'*x)[1] + T[1]
					if iprint >= 1
						println("lowold = ",lowold,"   low = ",lowerb," UB = ",UB)
					end
					if UB < GUB
						GUB = UB
						xUB = reshape(x,(n,1))
						upperb = (1-tol)*UB
						upperbmod = (1-tol)*(UB-T[1])
						if iprint >= 7
							println(lp)
							println("upperb = ",GUB)
							println("funzione obiettivo: ",0.5*(x'*Q*x) + (c'*x)[1] + T[1])
							println("veq: ",Aeq*x - beq)
							println("vin: ",A*x-b)
							println("x: ",x)
							println(raw_status(lp))
							#write_to_file(lp,"errore.mps")
							readline()
						end
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
				elseif flag == 2
					println("not feas (1)")
					break
				else
					println("flag = 3")
					lowerb = lowsafe
					break
				end
			else
				println("not feas (2)")
				break
			end
			if iprint >= 0
				@printf("                      lowerb-lowold:%15.8e\n",lowerb-lowold);
			end
			#if ((lowerb - lowold) <= max(    tol,0.1*tol*abs(GUB)))
			if ((lowerb - lowold) <= max(0.5*tol,0.01*(GUB-lowerb)))
				break
			end
			break
		end
		if ~feas
			flag = 2
		end
		solvetime = time()-start_time

		if (flag == 2)
			println("LB not feasible")
			e.LB = Inf
			#e.UB = Inf
			lowerb = Inf
		end
		return (lowerb, e.UB, solvetime)

	end

	function CPXcompute_low(lp,prob)
		n = prob.n
		p = length(prob.Dn)
		set_objective_function(lp, sum(prob.c[i]*lp[:x][i] for i=1:n) + 0.5*sum(prob.Dn[i]*lp[:g][i] for i=1:p)
			+ 0.5*sum(lp[:x][i]*lp[:x][j]*prob.Qpos[i,j] for i=1:n, j=1:n))
		set_objective_sense(lp, MOI.MIN_SENSE)
		#@objective(lp, Min, sum(prob.c[i]*lp[:x][i] for i=1:n) + 0.5*sum(prob.Dn[i]*lp[:g][i] for i=1:p)
		#	+ 0.5*sum(lp[:x][i]*lp[:x][j]*prob.Qpos[i,j] for i=1:n, j=1:n))

		try
			JuMP.optimize!(lp)
		catch
			println("CPLEX returned an error and did not solve the problem")
			flag = 3
			low = Nothing
			sol = []
			#readline()
		end
		prob.numlp = prob.numlp+1
		feas = true

		if(termination_status(lp) == MOI.OPTIMAL) || (termination_status(lp) == MOI.LOCALLY_SOLVED)
			low = objective_value(lp) + prob.T[1]
			sol = zeros(n+p+p)
			sol[prob.vmap["x"]] = copy(JuMP.value.(lp.obj_dict[:x]))
			sol[prob.vmap["z"]] = copy(JuMP.value.(lp.obj_dict[:z]))
			sol[prob.vmap["g"]] = copy(JuMP.value.(lp.obj_dict[:g]))
			flag = 1
		elseif(termination_status(lp) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing low")
			flag = 2
			low = Inf
			sol = []
		else
			#println("CPLEX: ",termination_status(lp))
			flag = 3
			low = Nothing
			sol = []
		end
		#println(sol)
		#@objective(lp, Min, sum(prob.c[i]*lp[:x][i] for i=1:n) )
		#println(lp)
		#println("paused in CPXcompute_low")
		#readline()
		return low, sol, flag

	end

	function IPOcompute_low(lp,prob)
		n = prob.n
		p = length(prob.Dn)
		set_objective_function(lp, sum(prob.c[i]*lp[:x][i] for i=1:n) + 0.5*sum(prob.Dn[i]*lp[:g][i] for i=1:p)
			+ 0.5*sum(lp[:x][i]*lp[:x][j]*prob.Qpos[i,j] for i=1:n, j=1:n))
		set_objective_sense(lp, MOI.MIN_SENSE)
		#@objective(lp, Min, sum(prob.c[i]*lp[:x][i] for i=1:n) + 0.5*sum(prob.Dn[i]*lp[:g][i] for i=1:p)
		#	+ 0.5*sum(lp[:x][i]*lp[:x][j]*prob.Qpos[i,j] for i=1:n, j=1:n))

		try
			JuMP.optimize!(lp)
		catch
			println("Ipopt returned an error and did not solve the problem")
			flag = 3
			low = Nothing
			sol = []
			#readline()
		end
		prob.numlp = prob.numlp+1
		feas = true

		if(termination_status(lp) == MOI.OPTIMAL) || (termination_status(lp) == MOI.LOCALLY_SOLVED)
			low = objective_value(lp) + prob.T[1]
			sol = zeros(n+p+p)
			sol[prob.vmap["x"]] = copy(JuMP.value.(lp[:x]))
			sol[prob.vmap["z"]] = copy(JuMP.value.(lp[:z]))
			sol[prob.vmap["g"]] = copy(JuMP.value.(lp[:g]))
			flag = 1
		elseif(termination_status(lp) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing low")
			flag = 2
			low = Inf
			sol = []
		else
			#println("IPOPT: ",termination_status(lp))
			flag = 3
			low = Nothing
			sol = []
		end
		#println(sol)
		#@objective(lp, Min, sum(prob.c[i]*lp[:x][i] for i=1:n) )
		#println(lp)
		#println("paused in CPXcompute_low")
		#readline()
		return low, sol, flag

	end

	function CPXfind_feasible_point(lp,prob)
			JuMP.optimize!(lp)
			prob.numlp = prob.numlp+1
			feas = true
			if(termination_status(lp) == MOI.OPTIMAL)
				sol = getvalue.(lp[:x])
				feas  = true
			else
				#println("\t infeasibility detected at root node !!!!!\n\n")
				sol = zeros(prob.n)
				feas  = false
			end
			#print(sol)
			return (reshape(sol,(prob.n,1)),feas)
	end

	function CPXcompute_max_z(lp,prob,i)
		@objective(lp,Max,lp[:z][i])
		JuMP.optimize!(lp)
		prob.numlp = prob.numlp+1
		feas = true

		if(termination_status(lp) == MOI.OPTIMAL)
			mx_i = objective_value(lp)
			flag = 1
		elseif(termination_status(lp) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing mx_zi for i =",i)
			flag = 2
			mx_i = 0.0
		else
			#write_model(lp,string("test_gp.lp"))
			#print("Hit Return to continue ...")
			#readline()
			mx_i = 0.0
			flag = 3
		end
		return mx_i, flag
	end

	function CPXcompute_min_z(lp,prob,i)
		@objective(lp,Min,lp[:z][i])
		#print(lp)
		#readline()
		JuMP.optimize!(lp)
		prob.numlp = prob.numlp+1
		feas = true

		if(termination_status(lp) == MOI.OPTIMAL)
			mx_i = objective_value(lp)
			flag = 1
		elseif(termination_status(lp) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing mn_zi for i =",i)
			flag = 2
			mx_i = 0.0
		else
			#write_model(lp,string("test_gp.lp"))
			#print("Hit Return to continue ...")
			#readline()
			mx_i = 0.0
			flag = 3
		end
		return mx_i, flag
	end

	function CPXbuild_initial_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!


		p = length(Dn)
		m,n   = size(A)
		meq,n = size(Aeq)
		n = length(c)

		lp = Model(CPLEX.Optimizer)
		set_optimizer_attribute(lp, "CPX_PARAM_SCRIND", CPX_OFF)
		@variable(lp, lowx[i] <= x[i = 1:n] <= upx[i])
		@variable(lp, lowz[i] <= z[i = 1:p] <= upz[i])
		@variable(lp,            g[i = 1:p] )
		@constraint(lp, stdcon[j=1:p], sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
		@constraint(lp, lineq[j=1:m], sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		@constraint(lp, leq[j=1:meq], sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		@constraint(lp, MC[i=1:p], -z[i]*(lowz[i]+upz[i]) + g[i] <= -lowz[i]*upz[i])

		#print(lp)
		#println("cpxbuild_model, hit return to continue")
		#readline()

		return lp
	end

	function check_feasibility(x,z,g,tolin,Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)
		#funzione che fa il check della feasibility

		tol = tolin

		p = length(Dn)
		m,n   = size(A)
		meq,n = size(Aeq)
		n = length(c)

		#check bound constraints
		viol = sum(x .>= upx .+ tol)
		if viol > 0
			#println("violation: ",viol)
			return 0
		end
		viol = sum(x .<= lowx .- tol)
		if viol > 0
			#println("violation: ",viol)
			return 0
		end
		viol = sum(z .>= upz .+ tol)
		if viol > 0
			#println("violation: ",viol)
			return 0
		end
		viol = sum(z .<= lowz .- tol)
		if viol > 0
			#println("violation: ",viol)
			return 0
		end
		for j = 1:p
			viol = abs(sum( Un[i,j]*x[i] for i=1:n ) - z[j])
			if viol >= tol
				#println("violation: ",viol)
				return 0
			end
			viol = -z[j]*(lowz[j]+upz[j]) + g[j] + lowz[j]*upz[j]
			if viol >= tol
				#println("violation: ",viol)
				return 0
			end
		end
		for j = 1:m
			viol = sum( A[j,i]*x[i] for i=1:n ) - b[j]
			if viol >= tol
				#println("violation: ",viol)
				return 0
			end
		end
		for j = 1:meq
			viol = abs(sum( Aeq[j,i]*x[i] for i=1:n ) - beq[j])
			if viol >= tol
				#println("violation: ",viol)
				return 0
			end
		end
		return 1
	end

	function CPXbuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!


		p = length(Dn)
		m,n   = size(A)
		meq,n = size(Aeq)
		n = length(c)

		lp = Model(CPLEX.Optimizer)
		set_optimizer_attribute(lp, "CPX_PARAM_SCRIND", CPX_OFF)
		set_optimizer_attribute(lp,"CPX_PARAM_BARQCPEPCOMP", 1.e-6)

		#lp = Model(Ipopt.Optimizer)
		#set_optimizer_attribute(lp, "print_level", 0)

		@variable(lp, lowx[i] <= x[i = 1:n] <= upx[i])
		@variable(lp, lowz[i] <= z[i = 1:p] <= upz[i])
		@variable(lp,            g[i = 1:p] >= -Inf)
		@objective(lp, Min, sum(c[i]*x[i] for i=1:n) + 0.5*sum(Dn[i]*g[i] for i=1:p)
			+ 0.5*sum(x[i]*x[j]*Qpos[i,j] for i=1:n, j=1:n))
		#@constraint(lp, stdcon[j=1:p], sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
		for j = 1:p
			@constraint(lp,  			sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
			@constraint(lp, 		    -z[j]*(lowz[j]+upz[j]) + g[j] <= -lowz[j]*upz[j])
		end
		#@constraint(lp, lineq[j=1:m], sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		for j = 1:m
			@constraint(lp, 		   sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		end
		#@constraint(lp, leq[j=1:meq], sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		for j = 1:meq
			@constraint(lp,  		   sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		end

		@constraint(lp, 		  sum(c[i]*x[i] for i=1:n) + 0.5*sum(Dn[i]*g[i] for i=1:p)
			+ 0.5*sum(x[i]*x[j]*Qpos[i,j] for i=1:n, j=1:n) <= upperboundmod)


		#print(lp)
		#println("cpxbuild_model, hit return to continue")
		#readline()

		return lp
	end

	function IPObuild_model(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod,tol)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!


		p = length(Dn)
		m,n   = size(A)
		meq,n = size(Aeq)
		n = length(c)

		#lp = Model(CPLEX.Optimizer)
		#set_optimizer_attribute(lp, "CPX_PARAM_SCRIND", CPX_OFF)
		#set_optimizer_attribute(lp,"CPX_PARAM_BARQCPEPCOMP", 1.e-5)
		lp = Model(Ipopt.Optimizer)
		set_optimizer_attribute(lp, "print_level", 0)
		set_optimizer_attribute(lp, "acceptable_constr_viol_tol", tol)

		@variable(lp, lowx[i] <= x[i = 1:n] <= upx[i])
		@variable(lp, lowz[i] <= z[i = 1:p] <= upz[i])
		@variable(lp,            g[i = 1:p] >= -Inf)
		@objective(lp, Min, sum(c[i]*x[i] for i=1:n) + 0.5*sum(Dn[i]*g[i] for i=1:p)
			+ 0.5*sum(x[i]*x[j]*Qpos[i,j] for i=1:n, j=1:n))
		#@constraint(lp, stdcon[j=1:p], sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
		for j = 1:p
			@constraint(lp,  			sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
			@constraint(lp, 		    -z[j]*(lowz[j]+upz[j]) + g[j] <= -lowz[j]*upz[j])
		end
		#@constraint(lp, lineq[j=1:m], sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		for j = 1:m
			@constraint(lp, 		   sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		end
		#@constraint(lp, leq[j=1:meq], sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		for j = 1:meq
			@constraint(lp,  		   sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		end

		@constraint(lp, 		  sum(c[i]*x[i] for i=1:n) + 0.5*sum(Dn[i]*g[i] for i=1:p)
			+ 0.5*sum(x[i]*x[j]*Qpos[i,j] for i=1:n, j=1:n) <= upperboundmod)

		#print(lp)
		#println("cpxbuild_model, hit return to continue")
		#readline()

		return lp
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

		(n,  h1) = size(S)
		(m,  h1) = size(A)
		(meq,h1) = size(Aeq)

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
