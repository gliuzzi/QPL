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

module partools_form2
	using Distributed
	using Gurobi

	export build_initial_model_2
	export build_model_2
	export set_lp_2
	export clear_lp_2
	export solutore_min_l_2
	export solutore_max_l_2
	export solutore_max_x_2

	export add_GUB_con_2!
	export add_MC_cons_2!
	export add_Qbound_cons_2!
	export set_matrices_2

	function __init__()
	     global env = Gurobi.Env()
	end

	n    = 0
	m    = 0
	nvar = 0
	Q    = 0
	c    = 0
	T    = 0
	A    = 0
	Aeq  = 0
	b    = 0
	beq  = 0
	vmap = 0
	lp   = 0

	function set_matrices_2(n_in,m_in,nvarin,Qin,cin,Tin,Ain,Aeqin,bin,beqin,vmapin)
		global n
		global m
		global nvar
		global Q
		global c
		global T
		global A
		global Aeq
		global b
		global beq
		global vmap
		global lp
		n = n_in
		m = m_in
		nvar = nvarin
		Q = copy(Qin)
		c = copy(cin)
		T = copy(Tin)
		A = copy(Ain)
		Aeq = copy(Aeqin)
		b = copy(bin)
		beq = copy(beqin)
		vmap = copy(vmapin)
		lp = Gurobi.Model(env,"init",:minimize)
	end

	function set_lp_2(low,up,lowx,upx,upperbound)
		global lp
		lp = build_model_2(Q,c,T,A,b,Aeq,beq,low,up,lowx,upx,upperbound)
	end

	function clear_lp_2(i)
		global lp
		Gurobi.free_model(lp)
	end

	function build_initial_model_2(Q,c,T,A,b,Aeq,beq,lowx,upx,upperbound)

		n = length(c)
		lp = Gurobi.Model(env, "QPL",:minimize);
		setparam!(lp,"OutputFlag",:false)

		add_variables_2!(lp,n,lowx,upx)
		add_standard_cons_2!(lp,n,upx,A,b,Aeq,beq)
		add_GUB_con_2!(lp,n,c,T[1],upperbound)

		return lp
	end

	function build_model_2(Q,c,T,A,b,Aeq,beq,low,up,lowx,upx,upperbound)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!

		#println("\t\tBuilding GUROBI model ...")
		n = length(c)
		lp = Gurobi.Model(env, "QPL",:minimize);
		setparam!(lp,"OutputFlag",:false)

		add_variables_2!(lp,n,lowx,upx)
		add_standard_cons_2!(lp,n,upx,A,b,Aeq,beq)
		add_GUB_con_2!(lp,n,c,T[1],upperbound)
		add_Qbound_cons_2!(lp,n,low,up,Q)
		add_MC_cons_2!(lp,n,low,up,upx,Q)

		#Gurobi.write_model(lp,"root.lp")
		#Gurobi.tune_model(lp)

		return lp
	end

	function add_variables_2!(lp,n,lowx,upx)
		#println("\t\t\tAdding x variables ...")
		add_cvars!(lp,zeros(n),reshape(lowx,(n,)),reshape(upx,(n,)))
		#println("\t\t\tAdding g variables ...")
		add_cvars!(lp,zeros(n),-Inf,Inf)
		#println("\t\t\t... done")
		update_model!(lp)
	end

	function add_standard_cons_2!(lp,n,upx,A,b,Aeq,beq)
		####################################################
		# Questa proc. aggiunge i vincoli:
		# A*x <= b, Aeq*x = beq
		####################################################
		Gurobi.set_dblattrarray!(lp,"UB",1,n,reshape(upx,(n,)))
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

	function add_GUB_con_2!(lp,n,c,const_,upperbound)
		####################################################
		# Questa proc. aggiunge il vincolo:
		# e'*g + c'x + const_ <= GUB
		####################################################
		#println("\t\t\tAdding constraint e'*g + c'*x <= GUB")
		#println("add_GUB_con!: ",const_)
		add_constr!(lp,[collect(1:n); collect((n+1):2n)],
				   [reshape(c,(n,));         ones(n)],'<',upperbound)
		update_model!(lp)
	end
	function add_Qbound_cons_2!(lp,n,low,up,Q)
		####################################################
		# Questa proc. aggiunge i vincoli:
		# Q[i,:]'*x >= low[i]  i = 1,...,n
		# Q[i,:]'*x <=  up[i]  i = 1,...,n
		####################################################
		for i = 1:n
			#println("\t\t\tAdding constraint  S[i,:]*x >= mn_li")
			add_constr!(lp,collect(1:n),Q[i,:],'>',low[i])
			#println("\t\t\tAdding constraint  S[i,:]*x <= mx_li")
			add_constr!(lp,collect(1:n),Q[i,:],'<',up[i])
		end
		update_model!(lp)
		#println("\t\t\t... done")
	end
	function add_MC_cons_2!(lp,n,low,up,upx,Q)
		####################################################
		# Questa proc. aggiunge i vincoli:
		# g_j >= 0.5*low[j]*x[j]   j = 1,...,n
		# g_j >= 0.5*upx[j]*S[j,:]*x + 0.5*up[j]*x[j] - 0.5*up[j]*upx[j]
		####################################################
		for j = 1:n
			e_j = zeros(n)
			e_j[j] = -1
			e_p = zeros(n)
			e_p[j] = low[j]
			#adding constraint f_j >= 0.5*mn_lj*x[j]
			add_constr!(lp,[collect(1:n);collect((n+1):2n)],[0.5*e_p;e_j],'<',0.0)
			e_p = upx[j]*Q[j,:]
			e_p[j] = e_p[j]+up[j]
			# adding constraint t_j >= 0.5mx_j * sum{k = 1..n}S[j,k]*x[k] + 0.5mx_lj*x[j] - 0.5mx_j*mx_lj
			add_constr!(lp,[collect(1:n);collect((n+1):2n)],[0.5*e_p;e_j],'<',0.5*up[j]*upx[j])
		end
		update_model!(lp)
	end

	function solutore_min_l_2(i)
		Gurobi.set_sense!(lp,:minimize)
		coefs = zeros(nvar)
		coefs[collect(vmap["x"])] = Q[i,:]
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
	    optimize(lp)
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

	function solutore_max_l_2(i)
		Gurobi.set_sense!(lp,:maximize)
		coefs = zeros(nvar)
		coefs[collect(vmap["x"])] = Q[i,:]
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
	    optimize(lp)
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mx_li = get_objval(lp)
			feas  = true
		else
			#println("\t infeasibility detected computing mn_li for i =",i)
			mx_li = 0.0
			feas  = false
		end
		return mx_li, feas
	end

	function solutore_max_x_2(i)
		Gurobi.set_sense!(lp,:maximize)
		coefs = zeros(nvar)
		coefs[collect(vmap["x"])[i]] = 1.0
		Gurobi.set_objcoeffs!(lp,coefs)
		Gurobi.update_model!(lp)
	    optimize(lp)
		feas = true
		if(Gurobi.status_symbols[Gurobi.GRB_OPTIMAL] == get_status(lp))
			mx_i = get_objval(lp)
			feas  = true
		else
			#println("\t infeasibility detected computing mn_li for i =",i)
			mx_i = 0.0
			feas  = false
		end
		return mx_i, feas
	end

end
