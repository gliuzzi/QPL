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

module BB_form2

	#using Utility
	using Printf
	using Gurobi

	import Base.<
	import Base.>
	import Base.<=
	import Base.copy
	import Base.push!
	import Base.show

	export BB_2
	export elem_bb_2
	export insert!, extract!, sort!

	export POLICY_VALUES, WHENLB_VALUES, BRANCH_VALUES

	POLICY_VALUES = Any[:lifo, :sort]
	WHENLB_VALUES = Any[:before, :after]
	BRANCH_VALUES = Any[:binary, :nary]

	#####################################################
	# an object of type "elem_bb_2" is a container defined by
	#	LB = a  lower bound
	#	UB = an upper bound
	#   x = minimum point corresponding to the LB
	#   g = minimum point corresponding to the LB
	#   index = index of x-variable onto which to branch
	#   lmin = lowerbound on the l function
	#   lmax = upperbound on the l function
	#   xmin = this should be equal to LB
	#   xmax = this should be equal to UB
	#   xUB
	#   lebel = level in the B&B tree
	#####################################################
	mutable struct elem_bb_2
		LB::Float64
		UB::Float64
		x::Array{Float64}
		g::Array{Float64}
		index::Int
		lmin::Array{Float64}
		lmax::Array{Float64}
		xmin::Array{Float64}
		xmax::Array{Float64}
		xUB::Array{Float64}
		level::Int

		function elem_bb_2(n)
			instance        = new()
			instance.LB     = -Inf
			instance.UB     = +Inf
			instance.xUB    = ones(n)/n
			instance.x      = copy(instance.xUB)
			instance.g      = zeros(n)
			instance.lmin   = zeros(n)
			instance.lmax   = Inf*ones(n)
			instance.xmin   = zeros(n)
			instance.xmax   = ones(n)
			instance.index  = 0
			instance.level  = 0

			return instance
		end

	end

	#####################################################
	# overload the copy operator
	#####################################################
	function copy(e::elem_bb_2)
		n = size(e.x,1)
 		a = elem_bb_2(n)
		a.LB   = copy(e.LB)
		a.UB   = copy(e.UB)
        a.x    = copy(e.x)
		a.g    = copy(e.g)
		a.lmin = copy(e.lmin)
		a.lmax = copy(e.lmax)
		a.xmin = copy(e.xmin)
		a.xmax = copy(e.xmax)
		a.xUB  = copy(e.xUB)
		a.level= e.level
		a.index= e.index
		return a
	end

	#####################################################
	# comparison function. Given two elem_bb_2 e1 and e2,
	# e1 > e2 iff e1.LB > e2.LB
	#####################################################
	function (>)(s1::elem_bb_2,s2::elem_bb_2)
		if(s1.LB > s2.LB)
			return true
		else
			return false
		end
	end
	#####################################################
	# comparison function. Given two elem_bb_2 e1 and e2,
	# e1 < e2 iff e1.LB < e2.LB
	#####################################################
	function (<)(s1::elem_bb_2,s2::elem_bb_2)
		if(s1.LB < s2.LB)
			return true
		else
			return false
		end
	end
	#####################################################
	# comparison function. Given two elem_bb_2 e1 and e2,
	# e1 <= e2 iff e1.LB <= e2.LB
	#####################################################
	function (<=)(s1::elem_bb_2,s2::elem_bb_2)
		if(s1.LB <= s2.LB)
			return true
		else
			return false
		end
	end

	mutable struct BB_2
		#####################################################
		# object implementing Branch&Bound
		#####################################################
		n::Int64				# number of original variables
		nvar::Int64				# number of variables in LP/MILP
		m::Int64				# number of ineq.constraints
		meq::Int64				# number of eq.constraints
		Q::Matrix               # hessian of the objective function
		c::Array{Float64}		# linear term of the objective function
		T::Array{Float64}		# constant term of the objective function
		A::Matrix               # Matrix of the ineq. constraints
		Aeq::Matrix             # Matrix of the eq. constraints
		b::Array{Float64}       # RHS of the ineq. constraints
		beq::Array{Float64}     # RHS of the eq. constraints
		nnodes::Int64			# number of open problems
		totnodes::Int64			# total number of generated subproblems
		GUB::Float64			# global upper bound i.e. estimate of the optimal solution
		GLB::Float64			# global lower bound i.e. worst lower bound among open nodes
		xUB::Array{Float64}		# point corresponding to the GUB value
		GAP::Float64			# optimality gap
		numlp::Int64			# number of LP's solved
		nummilp::Int64			# number of MILP's solved
		rebuild::Bool			# tells whether LP problems at the B&B tree nodes must be recomputed
								# from scratch every time an lb must be computed
		branch::Symbol			# kind of branch strategy. Values are {:binary, :nary}
		policy::Symbol			# how the queue is managed. Values are {:lifo, :sort}
		whenlb::Symbol			# when LB of node is computed, i.e. before or after node subdivision.
								# Possible values are {:before, :after}
		#####################################################
		# Array of open (sub)problems
		#####################################################
		open_probs
		vmap::Dict

		function BB_2()
			instance = new()
			instance.open_probs = Array{elem_bb_2}(undef,0)
			instance.nnodes = 0
			instance.totnodes = 0
			instance.n = -1
			instance.m = -1
			instance.Q = Array{Float64}(undef,0,0)
			instance.c = Array{Float64}(undef,0)
			instance.T = Array{Float64}(undef,0)
			instance.A = Array{Float64}(undef,0,0)
			instance.Aeq = Array{Float64}(undef,0,0)
			instance.b = Array{Float64}(undef,0)
			instance.beq = Array{Float64}(undef,0)
			instance.GUB = +Inf
			instance.GLB = -Inf
			instance.xUB = Array{Float64}(undef,0)
			instance.GAP = +Inf
			instance.rebuild = false
			instance.policy = :lifo
			instance.whenlb = :after
			instance.branch = :binary
			instance.numlp  = 0
			instance.nummilp= 0
			return instance
		end
	end


	#####################################################
	# insert subproblem in object
	#####################################################
	function insert!(prob::BB_2,e::elem_bb_2)
		(n,) = size(e.x)
		if (n==prob.n)
			ee = copy(e)
			push!(prob.open_probs,ee)
			prob.nnodes += 1
		else
			@printf("BB.insert!: WARNING: size of problem (%d) and size of given elem_bb_2 (%d) MISMATCH\n",prob.n,n)
		end
	end
	#####################################################
	# extract subproblem from object
	#####################################################
	function extract!(prob::BB_2)
		if(prob.nnodes > 0)
			prob.nnodes -= 1
			if((prob.policy == :lifo) || (prob.policy == :sort))
				return pop!(prob.open_probs)
			else
				return shift!(prob.open_probs)
			end
		else
			println("BB.extract!: WARNING: prob structure is EMPTY")
			return false
		end
	end
	#####################################################
	# sort open subproblems according to value of LB
	#####################################################
	function sort!(prob::BB_2, o::Function)
		Base.sort!(prob.open_probs,lt=(o))
	end
end
