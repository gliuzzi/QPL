__precompile__()

module utility
	using MAT
	using LinearAlgebra

	export read_problem
	export decompose_Q
	
	function decompose_Q(Q)
		n,nn = size(Q)
		#F = eigen(Q + 1.e-9*diagm(ones(n)))
		F = eigen(Q)
		U = F.vectors
		LAMBDA = F.values
		firstnneg = n+1
		#for i in 1:n
		#	if abs(LAMBDA[i]) < 1.e-9
		#		LAMBDA[i] = 0.0
		#	end
		#end
		for i in 1:n
			#if LAMBDA[i] >= 0.0
			if LAMBDA[i] >= -1.e-9
				firstnneg = i
				break
			end
		end
		if firstnneg <= n
			Qpos = zeros(n,n)
			for i in firstnneg:n
				Qpos += U[:,i]*U[:,i]'*LAMBDA[i]
			end
		#	Qpos[findall(x -> abs(x) < 1.e-7,Qpos)] .= 0.0
			Qpos = (Qpos+Qpos')/2
			Qpos = Qpos + 1.e-10*diagm(ones(n))
		else
			Qpos = Array{Float64,2}(undef,0,0)
		end
		#print(Qpos[1,:])
		for i in 1:n
			println("LAMBDA[",i,"] = ",LAMBDA[i])
		end
		#println("lambda=",LAMBDA)
		println("firstnneg=",firstnneg)
		println(LAMBDA[firstnneg])
		println(minimum(abs.(Qpos))," ",maximum(Qpos))
		println("eig=",eigen(Qpos).values)
		#readline()
		num_nneg = n - firstnneg + 1
		return Qpos,LAMBDA[1:firstnneg-1],U[:,1:firstnneg-1],num_nneg
	end
	
	function read_problem(probfile::String)
		###########################
		#    define:              #
		#    PROBLEM DIMENSION    #
		#    MATRIX  Q            #
		###########################

		println("\nReading problem data from file. ")
		println("file: ",probfile)

		fileid = matopen(probfile)
		conten = read(fileid)
		close(fileid)

		LB  = conten["LB"]
		UB  = conten["UB"]
		c   = conten["f"]
		Q   = conten["H"]
		A   = conten["A"]
		b   = conten["b"]
		Aeq = conten["Aeq"]
		beq = conten["beq"]
		n   = length(LB)
		#Q   = rand(n,n)
		Q   = (Q+Q')/2

		Qinit = copy(Q)
		cinit = copy(c)

		println("...done\n\n")
		println("Preprocess problem data to:")
		println("1) scale variables between 0 and 1")
		println("2) symmetrize Q matrix (in case it is not)")
		println("3) check for infinities in LB and/or UB")

		if minimum(LB) > -Inf && maximum(UB) < Inf
			#L <= x <= U
			#y = (x-L/(U-L)	x = L + (U-L)y
			#
			#0.5x'Qx + c'x = 0.5(L+(U-L)y)'Q(L+(U-L)y) + c'(L+(U-L)y) =
			#0.5L'QL + c'L + 0.5y'(U-L)Q(U-L)y + (L'Q + c')(U-L)y
			#
			#Qtilde = (U-L)Q(U-L)
			#ctilde = (U-L)(QL + c)
			#const  = 0.5L'QL + c'L
			#
			#0.5x'Qx + c'x = 0.5y'Qtilde y + ctilde' y + const

			#print([LB, UB])

			LAMBDA = diagm(reshape(UB-LB,(n,)))
			Qtilde = LAMBDA*Q*LAMBDA
			ctilde = LAMBDA*(Q*LB + c)
			const_ = 0.5*LB'*Q*LB + c'*LB

			################ PREPROCESS INPUT DATA ##############
			# minimo = zeros(n)
			# for i =1:n
			# 	#Qtilde[i,i] = Qtilde[i,i]+1000;
			# 	minimo[i]=minimum(Qtilde[i,:]);
			# end
			#
			# for i=1:n
			# 	for j=1:n
			# 		Qtilde[i,j]=Qtilde[i,j]-minimo[i];
			# 	end
			# end
			#
			# Qtilde = 0.5*(Qtilde+Qtilde')
			# ctilde = ctilde + minimo

			xtest = rand(n,1)
			println("prova 1:",0.5*xtest'*Qtilde*xtest + ctilde'*xtest + const_)
			println("prova 2:",0.5*(LB + xtest.*(UB-LB))'*Qinit*(LB+xtest.*(UB-LB)) + cinit'*xtest )
			################ PREPROCESS INPUT DATA END  #########

			Atilde = A*LAMBDA
			btilde = b - A*LB
			Aeqtil = Aeq*LAMBDA
			beqtil = beq - Aeq*LB
			Ltilde = zeros(n,1)
			Utilde = ones(n,1)
			flag_ok= true
		else
			Qtilde = Q
			ctilde = c
			const_ = reshape([0.0],(1,1))
			Atilde = A
			btilde = b
			Aeqtil = Aeq
			beqtil = beq
			Ltilde = LB
			Utilde = UB
			flag_ok= false
		end
		DATA = Dict(
			"LB"   => Ltilde,
			"UB"   => Utilde,
			"c"    => ctilde,
			"Q"    => Qtilde,
			"A"    => Atilde,
			"b"    => btilde,
			"Aeq"  => Aeqtil,
			"beq"  => beqtil,
			"T"    => const_,
			"flag" => flag_ok
		)
		return DATA
	end
end