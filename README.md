# QPL
 Repository of the B&B algorithm to solve  Nonconvex Quadratic problems with general linear constraints. N.B. the codes uses the external solvers <b>CPLEX</b>, <b>Gurobi</b> and <b>Ipopt</b> plus the <b>JuMP</b> package. The code has been tested with the following versions of the above solvers:
 - CPLEX v. 12.10
 - Gurobi v. 9.1.1
 - Ipopt v. 0.6.4
 - JuMP v. 0.21.5

Hence, prior of using the code you must install these solvers along with their respective Julia interfaces.

Codes have been developed and run on a Julia environment with the following packages installed:
```
julia> Pkg.status()
Status `C:\Users\giamp\.julia\environments\v1.5\Project.toml`
  [a076750e] CPLEX v0.7.3
  [0a46da34] CSDP v0.7.0
  [336ed68f] CSV v0.8.2
  [9961bab8] Cbc v0.7.1
  [a93c6f00] DataFrames v0.22.2
  [60bf3e95] GLPK v0.14.3
  [2e9cd046] Gurobi v0.8.0
  [b6b21f68] Ipopt v0.6.4
  [4076af6c] JuMP v0.21.5
  [e5e0dc1b] Juno v0.8.4
  [23992714] MAT v0.9.2
```

### How to run the code
All the codes must be run from within folder ```codes```. There are basically two ways of running the codes.
1. edit file main.jl to suit your needs. Then, from a command prompt execute
```
$ julia -p <nproc> main.jl
```
2. edit file main1.jl to suit your needs. Then, from a julia REPL execute
```
julia> include("main1.jl")
```

### Folders description
 1. randqp - this is were random QP instances are stored. Problems are provided as .mat Matlab&reg; files
 2. codes - this is were the codes are stored
 3. test - this is were scripts to reproduce the results of the paper are stored

# Publication
G.Liuzzi, M.Locatelli, V.Piccialli "A computational study on QP problems with general linear constraints". Submitted to Optimization Letters (2021)

##### AUTORI: G. Liuzzi<sup>1</sup>, M. Locatelli<sup>2</sup>, V. Piccialli<sup>3</sup>

 <sup>1</sup> Department of Computer, Control and Management Engineering, "Sapienza" University of Rome (giampaolo.liuzzi@uniroma1.it)

 <sup>2</sup> Department of Engineering and Architecture, University of Parma (marco.locatelli@unipr.it)

 <sup>3</sup> Department of Civil Engineering and Computer Science Engineering, University of Rome "Tor Vergata" (veronica.piccialli@uniroma2.it)


Copyright 2021
