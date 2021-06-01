__precompile__()

module partools_form10
	using Distributed
	using JuMP
	using CPLEX

	#export CPXbuild_model
	export CPXsolutore_max_z_10
	export CPXsolutore_min_z_10
	
	export set_matrices_10
	export clear_matrices_10
	export set_lp_10
	export clear_lp_10
	export garbage_collection

	n    = 0
	p    = 0
	m    = 0
	meq  = 0
	nvar = 0
	Q    = 0
	Qpos = 0
	Dn   = 0
	Un   = 0
	c    = 0
	T    = 0
	A    = 0
	Aeq  = 0
	b    = 0
	beq  = 0
	vmap = 0
	lpp  = 0
	lpmax = 0
	lpmin = 0

	function garbage_collection(i)
		GC.gc()
	end
	function clear_matrices_10(i)
		global n
		global p
		global m
		global meq
		global nvar
		global Q
		global Qpos
		global Dn
		global Un
		global c
		global T
		global A
		global Aeq
		global b
		global beq
		global vmap

		n    = 0
		p    = 0
		m    = 0
		meq  = 0
		nvar = 0
		Q    = 0
		Qpos = 0
		Dn   = 0
		Un   = 0
		c    = 0
		T    = 0
		A    = 0
		Aeq  = 0
		b    = 0
		beq  = 0
		vmap = 0

	end 

	function clear_lp_10(i)
		global lpmax
		global lpmin

		try
			empty!(lpmax)
		catch
		end
		try
			empty!(lpmin)
		catch
		end
		lpmax = 0
		lpmin = 0
	end

	function set_matrices_10(n_in,p_in,m_in,meq_in,nvarin,Qin,Qposin,Dnin,Unin,cin,Tin,Ain,Aeqin,bin,beqin,vmapin)
		global n
		global p
		global m
		global meq
		global nvar
		global Q
		global Qpos
		global Dn
		global Un
		global c
		global T
		global A
		global Aeq
		global b
		global beq
		global vmap
		global lpmax
		global lpmin
		n = n_in
		p = p_in
		m = m_in
		meq = meq_in
		nvar = nvarin
		Q = copy(Qin)
		Qpos = copy(Qposin)
		Dn= copy(Dnin)
		Un= copy(Unin)
		c = copy(cin)
		T = copy(Tin)
		A = copy(Ain)
		Aeq = copy(Aeqin)
		b = copy(bin)
		beq = copy(beqin)
		vmap = copy(vmapin)
	end

	function set_lp_10(lowx,upx,lowz,upz,upperbound)
		global lpmax
		global lpmin
		lpmax = CPXbuild_model_worker(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperbound)
		lpmin = CPXbuild_model_worker(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperbound)
	end

	function CPXbuild_model_worker(Q,Qpos,Dn,Un,c,T,A,b,Aeq,beq,lowx,upx,lowz,upz,upperboundmod)

		#N.B. la funzione ob. cambia ogni volta che selezioniamo il problema
		#     quindi è inizializzata a zero!


		n = length(c)
		p = length(Dn)
		m,~   = size(A)
		meq,~ = size(Aeq)

		lp = Model(CPLEX.Optimizer)
		set_optimizer_attribute(lp, "CPX_PARAM_SCRIND", CPX_OFF)
		set_optimizer_attribute(lp,"CPX_PARAM_BARQCPEPCOMP", 1.e-6)
		@variable(lp, lowx[i] <= x[i = 1:n] <= upx[i]) 
		@variable(lp, lowz[i] <= z[i = 1:p] <= upz[i]) 
		@variable(lp,            g[i = 1:p] ) #>= 0.0) 
		@constraint(lp, stdcon[j=1:p], sum( Un[i,j]*x[i] for i=1:n ) == z[j] )
		@constraint(lp, lineq[j=1:m], sum( A[j,i]*x[i] for i=1:n ) <= b[j] )
		@constraint(lp, leq[j=1:meq], sum( Aeq[j,i]*x[i] for i=1:n ) == beq[j] )
		@constraint(lp, GUB_con, sum(c[i]*x[i] for i=1:n) + 0.5*sum(Dn[i]*g[i] for i=1:p) 
			+ 0.5*sum(x[i]*x[j]*Qpos[i,j] for i=1:n, j=1:n) <= upperboundmod)
		@constraint(lp, MC[i=1:p], -z[i]*(lowz[i]+upz[i]) + g[i] <= -lowz[i]*upz[i])

		#print(lp)
		#println("cpxbuild_model, hit return to continue")
		#readline()
		
		return lp
	end

	function CPXsolutore_max_z_10(i)
		@objective(lpmax,Max,lpmax[:z][i])
		JuMP.optimize!(lpmax)

		if(termination_status(lpmax) == MOI.OPTIMAL)
			mx_i = objective_value(lpmax)
			flag = 1
		elseif(termination_status(lpmax) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing mx_zi for i =",i)
			flag = 2
			mx_i = -Inf
		else
			mx_i = -Inf
			flag = 3
		end
		return mx_i, flag
	end

	function CPXsolutore_min_z_10(i)
		@objective(lpmin,Min,lpmin[:z][i])
		JuMP.optimize!(lpmin)

		if(termination_status(lpmin) == MOI.OPTIMAL)
			mx_i = objective_value(lpmin)
			flag = 1
		elseif(termination_status(lpmin) == MOI.INFEASIBLE)
			#println("\t infeasibility detected computing mn_zi for i =",i)
			flag = 2
			mx_i = Inf
		else
			mx_i = Inf
			flag = 3
		end
		return mx_i, flag
	end

end
