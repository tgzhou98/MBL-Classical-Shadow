using ITensors
using LinearAlgebra

####################################################################################
# For calculation of ||OA||^2 Shadow norm
# To construct the operator in the DOUBLE OPERATOR space
q = 2               # qubit dimension
paulidim = q^2      # operator dimension
# Define the new double operator space
ITensors.space(::SiteType"doubleOpState") = paulidim^2  # Double operator dimension

# Auxilary operator in the single operator space
IdL_mat = [1	0	0	0
0	1	0	0
0	0	1	0
0	0	0	1]
SzL_mat = [0	0	0	1
0	0	1im	0
0	-1im	0	0
1	0	0	0]
SpL_mat = [0	1	1im	0
1	0	0	1
1im	0	0	1im
0	-1	-1im	0]
SmL_mat = [0	1	-1im	0
1	0	0	-1
-1im	0	0	1im
0	1	-1im	0]
SzR_mat = ([0	0	0	-1
0	0	1im	0
0	-1im	0	0
-1	0	0	0])
SpR_mat = ([0	-1	-1im	0
-1	0	0	1
-1im	0	0	1im
0	-1	-1im	0])
SmR_mat = ([0	-1	1im	0
    -1	0	0	-1
    1im	0	0	1im
    0	1	-1im	0])
IdR_mat =  [1	0	0	0
0	1	0	0
0	0	1	0
0	0	0	1]

# Superoperator in the double operator space
ITensors.op(::OpName"IdL1",::SiteType"doubleOpState") = kron(IdL_mat, IdL_mat)

ITensors.op(::OpName"SzL1",::SiteType"doubleOpState") = kron(SzL_mat, IdL_mat)
    
ITensors.op(::OpName"S+L1",::SiteType"doubleOpState") = kron(SpL_mat, IdL_mat)

ITensors.op(::OpName"S-L1",::SiteType"doubleOpState") = kron(SmL_mat, IdL_mat)

ITensors.op(::OpName"IdR1",::SiteType"doubleOpState") = kron((IdR_mat), IdL_mat)
                                                                   
ITensors.op(::OpName"SzR1",::SiteType"doubleOpState") = kron((SzR_mat), IdL_mat)
                                                                   
ITensors.op(::OpName"S+R1",::SiteType"doubleOpState") = kron((SpR_mat), IdL_mat)
                                                                   
ITensors.op(::OpName"S-R1",::SiteType"doubleOpState") = kron((SmR_mat), IdL_mat)

ITensors.op(::OpName"IdL2",::SiteType"doubleOpState") = kron(IdL_mat, IdL_mat)
                                                                                
ITensors.op(::OpName"SzL2",::SiteType"doubleOpState") = kron(IdL_mat, (SzL_mat))
                                                                            
ITensors.op(::OpName"S+L2",::SiteType"doubleOpState") = kron(IdL_mat, (SpL_mat))
                                                                            
ITensors.op(::OpName"S-L2",::SiteType"doubleOpState") = kron(IdL_mat, (SmL_mat))

ITensors.op(::OpName"IdR2",::SiteType"doubleOpState") = kron(IdL_mat, (IdR_mat))
                                                                          
ITensors.op(::OpName"SzR2",::SiteType"doubleOpState") = kron(IdL_mat, (SzR_mat))

ITensors.op(::OpName"S+R2",::SiteType"doubleOpState") = kron(IdL_mat, (SpR_mat))

ITensors.op(::OpName"S-R2",::SiteType"doubleOpState") = kron(IdL_mat, (SmR_mat))


# The state 
ITensors.state(::StateName"IIstate", ::SiteType"doubleOpState") = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"Occstate", ::SiteType"doubleOpState") = [0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 1/3]
ITensors.state(::StateName"initstate", ::SiteType"doubleOpState") = [1, 0, 0, 0, 0, 1/(1 + q), 0, 0, 0, 0, 1/(1 + q), 0, 0, 0, 0, 1/(1 + q)]
####################################################################################

####################################################################################
# For sampling and reconstruction stages
# Single site random Clifford gate in the original basis

# The raw data are load from the mathematica calculation
clifford_rep_arr_raw = [[[1, 0], [0, 1]], [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -(1/sqrt(2))]], [[1, 0], [0, 1im]], [[1, 0], [0, -1]], [[1, 
0], [0, -1im]], [[0, 1], [1, 0]], [[0, -1im], [1im, 0]], [[1/sqrt(2), 1im/
sqrt(2)], [1/sqrt(2), -(1im/sqrt(2))]], [[1/sqrt(2), -(1/sqrt(2))], [1/sqrt(2), 1/sqrt(2)]], [[1/sqrt(2), -(1im/sqrt(2))], [1/sqrt(2), 1im/sqrt(2)]], [[1/sqrt(2), 1/sqrt(2)], [-(1/sqrt(2)), 1/sqrt(2)]], [[1im/sqrt(2), -(1im/sqrt(2))], [-(1im/sqrt(2)), -(1im/sqrt(2))]], [[1/sqrt(2), 1/
sqrt(2)], [1im/sqrt(2), -(1im/sqrt(2))]], [[0, 1], [1im, 
0]], [[0, -1im], [-1, 0]], [[1/sqrt(2), 1/sqrt(2)], [-(1im/sqrt(2)), 1im/
sqrt(2)]], [[1/2 + 1im/2, 1/2 - 1im/2], [1/2 - 1im/2, 1/2 + 1im/2]], [[1im/
sqrt(2), 1/sqrt(2)], [-(1im/sqrt(2)), 1/sqrt(2)]], [[-(1/sqrt(2)), -(1im/sqrt(2))], [1/sqrt(2), -(1im/sqrt(2))]], [[1/2 - 1im/2, 1/2 + 1im/2], [1/2 + 1im/2, 
1/2 - 1im/2]], [[1/sqrt(2), -(1/sqrt(2))], [1im/sqrt(2), 1im/sqrt(2)]], [[1/sqrt(2), -(1im/sqrt(2))], [1im/sqrt(2), -(1/sqrt(2))]], [[1im/
sqrt(2), -(1im/sqrt(2))], [1/sqrt(2), 1/sqrt(2)]], [[1/sqrt(2), 1im/
sqrt(2)], [-(1im/sqrt(2)), -(1/sqrt(2))]]]

clifford_rep_arr = Array{ComplexF64,2}[]
for i in 1:length(clifford_rep_arr_raw)
    push!(clifford_rep_arr, hcat(clifford_rep_arr_raw[i][1], clifford_rep_arr_raw[i][2]))
end
clif_num = length(clifford_rep_arr)



# Define the 24 operator and the operators in dagger
ITensors.op(::OpName"cli1",::SiteType"S=1/2") = clifford_rep_arr[1]
ITensors.op(::OpName"cli2",::SiteType"S=1/2") = clifford_rep_arr[2]
ITensors.op(::OpName"cli3",::SiteType"S=1/2") = clifford_rep_arr[3]
ITensors.op(::OpName"cli4",::SiteType"S=1/2") = clifford_rep_arr[4]
ITensors.op(::OpName"cli5",::SiteType"S=1/2") = clifford_rep_arr[5]
ITensors.op(::OpName"cli6",::SiteType"S=1/2") = clifford_rep_arr[6]
ITensors.op(::OpName"cli7",::SiteType"S=1/2") = clifford_rep_arr[7]
ITensors.op(::OpName"cli8",::SiteType"S=1/2") = clifford_rep_arr[8]
ITensors.op(::OpName"cli9",::SiteType"S=1/2") = clifford_rep_arr[9]
ITensors.op(::OpName"cli10",::SiteType"S=1/2") = clifford_rep_arr[10]
ITensors.op(::OpName"cli11",::SiteType"S=1/2") = clifford_rep_arr[11]
ITensors.op(::OpName"cli12",::SiteType"S=1/2") = clifford_rep_arr[12]
ITensors.op(::OpName"cli13",::SiteType"S=1/2") = clifford_rep_arr[13]
ITensors.op(::OpName"cli14",::SiteType"S=1/2") = clifford_rep_arr[14]
ITensors.op(::OpName"cli15",::SiteType"S=1/2") = clifford_rep_arr[15]
ITensors.op(::OpName"cli16",::SiteType"S=1/2") = clifford_rep_arr[16]
ITensors.op(::OpName"cli17",::SiteType"S=1/2") = clifford_rep_arr[17]
ITensors.op(::OpName"cli18",::SiteType"S=1/2") = clifford_rep_arr[18]
ITensors.op(::OpName"cli19",::SiteType"S=1/2") = clifford_rep_arr[19]
ITensors.op(::OpName"cli20",::SiteType"S=1/2") = clifford_rep_arr[20]
ITensors.op(::OpName"cli21",::SiteType"S=1/2") = clifford_rep_arr[21]
ITensors.op(::OpName"cli22",::SiteType"S=1/2") = clifford_rep_arr[22]
ITensors.op(::OpName"cli23",::SiteType"S=1/2") = clifford_rep_arr[23]
ITensors.op(::OpName"cli24",::SiteType"S=1/2") = clifford_rep_arr[24]


ITensors.op(::OpName"dagcli1",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[1] )
ITensors.op(::OpName"dagcli2",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[2] )
ITensors.op(::OpName"dagcli3",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[3] )
ITensors.op(::OpName"dagcli4",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[4] )
ITensors.op(::OpName"dagcli5",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[5] )
ITensors.op(::OpName"dagcli6",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[6] )
ITensors.op(::OpName"dagcli7",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[7] )
ITensors.op(::OpName"dagcli8",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[8] )
ITensors.op(::OpName"dagcli9",::SiteType"S=1/2")  = adjoint(clifford_rep_arr[9] )
ITensors.op(::OpName"dagcli10",::SiteType"S=1/2") = adjoint(clifford_rep_arr[10])
ITensors.op(::OpName"dagcli11",::SiteType"S=1/2") = adjoint(clifford_rep_arr[11])
ITensors.op(::OpName"dagcli12",::SiteType"S=1/2") = adjoint(clifford_rep_arr[12])
ITensors.op(::OpName"dagcli13",::SiteType"S=1/2") = adjoint(clifford_rep_arr[13])
ITensors.op(::OpName"dagcli14",::SiteType"S=1/2") = adjoint(clifford_rep_arr[14])
ITensors.op(::OpName"dagcli15",::SiteType"S=1/2") = adjoint(clifford_rep_arr[15])
ITensors.op(::OpName"dagcli16",::SiteType"S=1/2") = adjoint(clifford_rep_arr[16])
ITensors.op(::OpName"dagcli17",::SiteType"S=1/2") = adjoint(clifford_rep_arr[17])
ITensors.op(::OpName"dagcli18",::SiteType"S=1/2") = adjoint(clifford_rep_arr[18])
ITensors.op(::OpName"dagcli19",::SiteType"S=1/2") = adjoint(clifford_rep_arr[19])
ITensors.op(::OpName"dagcli20",::SiteType"S=1/2") = adjoint(clifford_rep_arr[20])
ITensors.op(::OpName"dagcli21",::SiteType"S=1/2") = adjoint(clifford_rep_arr[21])
ITensors.op(::OpName"dagcli22",::SiteType"S=1/2") = adjoint(clifford_rep_arr[22])
ITensors.op(::OpName"dagcli23",::SiteType"S=1/2") = adjoint(clifford_rep_arr[23])
ITensors.op(::OpName"dagcli24",::SiteType"S=1/2") = adjoint(clifford_rep_arr[24])
####################################################################################