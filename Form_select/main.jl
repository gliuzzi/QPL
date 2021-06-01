#####################################################
using Distributed
@everywhere push!(LOAD_PATH, pwd())
@everywhere using partools_form2

using Printf
using mod_run

nloc       = 10
MAXNODES   = Inf
TOLGAP     = 1.e-5
time_limit = 1800

if true
	res   = open("results.txt","a")
	@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
	@printf(res,  "           use always FORM(2)        \n")
	@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
	@printf(res,  "                 con strong Bound tightening su tutti i nodi\n")
	@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
	@printf(res,  "PROBLEM         |                 |    nneg |     GAP |             GUB |  NODES |    TIME\n")
	@printf(res,  "----------------+-----------------+---------+---------+-----------------+--------+---------\n")
	close(res)
end

triplets = [(20,1,3), (20,1,4), (20,2,1), 
			(20,3,2), (20,3,3), (20,3,4),
			(20,4,3), (20,4,4),
			(30,2,3), (30,2,4), (30,3,2),
			(30,3,4), (30,4,3), (30,4,4), (40,1,2), (40,1,3), (40,1,4), (40,2,2),
			(40,3,2), (40,3,3), (40,3,4), (40,4,1), (50,1,2), (50,1,4),
			(50,3,2), (50,3,4), (50,4,1), (50,4,2), (50,4,4)]

dims = [20, 30, 40, 50]
#dims = [50, 50, 50, 50, 50, 50, 50]
#dims = [20,30]

triplets = [         (20,3,2),(20,3,3),(20,3,4)]
triplets = [(20,3,1),(20,3,2),(20,3,3),(20,3,4)]

triplets = [(40,4,2), (40,4,3), (40,4,4),
			(50,1,1), (50,1,2), (50,1,3), (50,1,4),
			(50,2,1), (50,2,2), (50,2,3), (50,2,4),
			(50,3,1), (50,3,2), (50,3,3), (50,3,4),
			(50,4,1), (50,4,2), (50,4,3), (50,4,4)]

#for trip in triplets
for n in dims
	for i1 in 1:4
		for i2 in 1:4
			trip = (n,i1,i2)
			addprocs(10)
			@everywhere push!(LOAD_PATH, pwd())
			@everywhere using partools_form2
			warmup(nloc,MAXNODES,TOLGAP,time_limit)
			run_codes(nloc,MAXNODES,TOLGAP,time_limit,[trip])
			rmprocs(workers(),waitfor=0)
		end
	end
end

println("Fine")
nothing
