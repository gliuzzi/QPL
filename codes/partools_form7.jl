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

module partools_form7
	using Distributed
	using Gurobi

	export build_initial_model_7
	export build_model_7
	export set_lp_7
	export clear_lp_7
	export solutore_max_x_7
	export solutore_min_x_7

	export add_GUB_con_7!
	export add_MC_cons_7!
	export add_Qbound_cons_7!
	export set_matrices_7

	function __init__()
	     global env = Gurobi.Env()
	end

	n    = 0
	m    = 0
	nvar = 0
	Q    = 0
	Qvec = 0
	c    = 0
	T    = 0
	A    = 0
	Aeq  = 0
	b    = 0
	beq  = 0
	vmap = 0
	lp   = 0

	function set_matrices_7(n_in,m_in,nvarin,Qin,Qvecin,cin,Tin,Ain,Aeqin,bin,beqin,vmapin)
		global n
		global m
		global nvar
		global Q
		global Qvec
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
		Qvec = copy(Qvecin)
		c = copy(cin)
		T = copy(Tin)
		A = copy(Ain)
		Aeq = copy(Aeqin)
		b = copy(bin)
		beq = copy(beqin)
		vmap = copy(vmapin)
		lp = Gurobi.Model(env,"init",:minimize)
	end

	function set_lp_7(lowx,upx,upperbound)
		global lp
		lp = build_model_7(Q,Qvec,c,T,A,b,Aeq,beq,lowx,upx,upperbound)
	end

	function clear_lp_7(i)
		global lp
		Gurobi.free_model(lp)
	end

	function build_initial_model_7(Q,Qvec,c,T,A,b,Aeq,beq,lowx,upx,upperboundmod)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!

		#println("\t\tBuilding GUROBI model ...")
		n = length(c)
		lp = Gurobi.Model(env, "QPL",:minimize);
		setparam!(lp,"OutputFlag",:false)

		add_variables_7!(lp,n,lowx,upx)
		add_standard_cons_7!(lp,n,lowx,upx,A,b,Aeq,beq)
		add_GUB_con_7!(lp,n,Qvec,c,upperboundmod)
		return lp
	end

	function build_model_7(Q,Qvec,c,T,A,b,Aeq,beq,lowx,upx,upperboundmod)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!

		#println("\t\tBuilding GUROBI model ...")
		n = length(c)
		lp = Gurobi.Model(env, "QPL",:minimize);
		setparam!(lp,"OutputFlag",:false)

		add_variables_7!(lp,n,lowx,upx)
		add_standard_cons_7!(lp,n,lowx,upx,A,b,Aeq,beq)
		add_GUB_con_7!(lp,n,Qvec,c,upperboundmod)
		add_MC_cons_7!(lp,n,lowx,upx,Q)

		return lp
	end

	function add_variables_7!(lp,n,lowx,upx)
		#println("\t\t\tAdding x variables ...")
		add_cvars!(lp,zeros(n),reshape(lowx,(n,)),reshape(upx,(n,)))
		#println("\t\t\tAdding g variables ...")
		p = convert(Int,n*(n+1)/2)
		add_cvars!(lp,zeros(p),zeros(p),ones(p))
		#println("\t\t\t... done")
		update_model!(lp)
	end

	function add_standard_cons_7!(lp,n,lowx,upx,A,b,Aeq,beq)
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

	function add_GUB_con_7!(lp,n,Qvec,c,upperboundmod)
		####################################################
		# Questa proc. aggiunge il vincolo:
		# e'*g + c'x + const_ <= GUB
		####################################################
		#println("\t\t\tAdding constraint e'*g + c'*x <= GUB")
		#println("add_GUB_con!: ",const_)
		p       = convert(Int,n*(n+1)/2)
		add_constr!(lp,[collect(1:n); collect((n+1):(n+p))],
				   [reshape(c,(n,));         Qvec],'<',upperboundmod)
		update_model!(lp)
	end
	function add_MC_cons_7!(lp,n,lowx,upx,Q)
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

	function solutore_max_x_7(i)
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

	function solutore_min_x_7(i)
		Gurobi.set_sense!(lp,:minimize)
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
