
using Plots
using LaTeXStrings
using JLD2
using Statistics, LinearAlgebra
using Printf
using Distributions, Random
using Test
using Dates
using SparseArrays
# using Pkg ; Pkg.add("Distributions")

PauliMatrices = [[0 1; 1 0], [0 -1im; 1im 0], [1 0; 0 -1]] 
Idmatrix = [1 0; 0 1] .+ 0.0im
# kron(PauliMatrices[1], PauliMatrices[1])
⊗(x, y) = kron(x, y)

# Generate single Pauli operator in the whole Hilbert space (2^nsys_spin)
function gen_singlePauli(nsite::Int64,nsys_spin::Int64, sigma::Int64)
    Jij_term_ops = fill(Idmatrix, nsys_spin)
    Jij_term_ops[nsite] = PauliMatrices[sigma]

    H = foldl(⊗, Jij_term_ops)
    return sparse(H)
end 

# Input: a number nbasis in the range [0,2^nsys_spin-1]
# Ouput: an Integer array that denotes the basis
function basis_label_spin(nbasis::Int64,nsys_spin::Int64)
    # This function returns the Sz on each site. The input basis label is denoted as nbasis. Here 0 denotes Sz = -1, 1 denotes Sz=1.
    ns=nbasis
    basis_number_init = zeros(Int64, nsys_spin)
    for i in 1:nsys_spin
        basis_number_init[i]=rem(ns,4)
        ns=fld(ns,4)
    end
    basis_number = deepcopy(basis_number_init[:])
    return basis_number
end 

function gen_MBL(nsys_spin::Int64, W_disorder::Float64, Delta::Float64, spin_single_site_basis_arr::Vector{SparseMatrixCSC{ComplexF64, Int64}})
    # This function generates the spin SYK4 model in the spin 1/2 representation. nsys_spin is the number of spin 1/2.
    ndim = 2^nsys_spin
    ham = zeros(ComplexF64, ndim,ndim)
    
    h_arr = rand(Uniform(-W_disorder, W_disorder), nsys_spin)
    anisotropic_arr = [1, 1, Delta]
   
    for i in 1:nsys_spin - 1
        j = i + 1
        for a in 1:3
            anisotropic_arr 
            h1 = spin_single_site_basis_arr[i + nsys_spin*(a-1)]
            h2 = spin_single_site_basis_arr[j + nsys_spin*(a-1)]

            ham += 1 / 4 * h1 * h2 * anisotropic_arr[a]
        end
    end
    for i in 1:nsys_spin
        a = 3
        h1 = spin_single_site_basis_arr[i + nsys_spin*(a-1)]
        ham -= 1/ 2 * h1 * h_arr[i]
    end

    @test norm(ham - adjoint(ham)) == 0.0

    return ham
end 

function prepare_spin_basis(nsys_spin::Int64, data_dir::String, load_flag::Bool)
    if load_flag
        spin_single_site_basis_arr = JLD2.load(joinpath(data_dir, "Size_N$(nsys_spin)_spin_single_site_basis.jld")
            , "spin_single_site_basis_arr") 
        spin_hilbert_basis_arr = JLD2.load(joinpath(data_dir, "Size_N$(nsys_spin)_spin_hilbert_basis.jld")
            , "spin_hilbert_basis_arr")
    else
        ndim = 2^nsys_spin
        spin_single_site_basis_arr = vcat(
             [gen_singlePauli(i, nsys_spin, 1) for i in 1:nsys_spin]
            ,[gen_singlePauli(i, nsys_spin, 2) for i in 1:nsys_spin]
            ,[gen_singlePauli(i, nsys_spin, 3) for i in 1:nsys_spin]
        )
        JLD2.save(joinpath(data_dir, "Size_N$(nsys_spin)_spin_single_site_basis.jld")
            , "spin_single_site_basis_arr",spin_single_site_basis_arr)  
        spin1 = spin_single_site_basis_arr[1]
        each_spin_memory = Base.summarysize(spin1)

        spin_hilbert_basis_arr =  Vector{SparseMatrixCSC{ComplexF64, Int64}}()
        println("Easimate memory is $(2^(2*nsys_spin)*each_spin_memory/1024/1024/1024) Gbyte")
        for l in 0:4^(nsys_spin)-1
            if mod(l,20000) == 1
                println("Basis index is $(l+1) in $(2^(2*nsys_spin))")
                println("Current time is $(now())")
            end
            # dot(tempa, tempb) == tr(conj(tempa) * tempb)
            pauli_number_basis = basis_label_spin(l, nsys_spin)
            # spin_number = sum(pauli_number_basis)
            temp_basis_op = sparse(Matrix((1.0+0.0im)I, ndim, ndim))
            for site in 1:nsys_spin
                if pauli_number_basis[site] != 0
                    # temp_basis_op *= spin_single_site_basis_arr[site]
                    puali_type = pauli_number_basis[site]
                    temp_basis_op *= spin_single_site_basis_arr[site + nsys_spin*(puali_type-1)]
                end
            end
            append!(spin_hilbert_basis_arr, [temp_basis_op])
        end
        JLD2.save(joinpath(data_dir, "Size_N$(nsys_spin)_spin_hilbert_basis.jld")
            , "spin_hilbert_basis_arr", spin_hilbert_basis_arr)  
        @test dot(spin_hilbert_basis_arr[2], spin_hilbert_basis_arr[2])/ndim == 1.0
    end
    
    return spin_hilbert_basis_arr, spin_single_site_basis_arr
end


function sizeDis_windsizeDis(nsys_spin::Int64, W_disorder::Float64, Delta::Float64, t_list::Array{ComplexF64,1}, Nrealization::Int64, 
    spin_hilbert_basis_arr::Vector{SparseMatrixCSC{ComplexF64, Int64}},
    spin_single_site_basis_arr::Vector{SparseMatrixCSC{ComplexF64, Int64}}, OA_type::Int64)

    ndim = 2^nsys_spin
    Nt= length(t_list)
    targetstate = [0 for n=1:nsys_spin]
    k = nsys_spin÷2
    println("N=$(nsys_spin),k=$(k)")
    nonId_op_arr = collect(nsys_spin÷2-k÷2+1:nsys_spin÷2+k÷2)
    for i in nonId_op_arr 
        targetstate[i] = OA_type
    end
    display(targetstate)

    pauli_OA = sparse(Matrix((1.0+0.0im)I, ndim, ndim))
    for site in 1:nsys_spin
        if targetstate[site] != 0
            puali_type = targetstate[site]
            pauli_OA *= spin_single_site_basis_arr[site + nsys_spin*(puali_type-1)]
        end
    end

    c_basis_arr = zeros(ComplexF64, 4^(nsys_spin), Nt, Nrealization)
    Pn_arr = zeros(ComplexF64, nsys_spin+1, Nt, Nrealization)
    Qn_arr = zeros(ComplexF64, nsys_spin+1, Nt, Nrealization)

    for Nreal in 1:Nrealization
        println("Nrealization is $(Nreal)")
        ham = gen_MBL(nsys_spin, W_disorder, Delta, spin_single_site_basis_arr)
        norm_factor = sqrt( tr(pauli_OA * pauli_OA) / ndim )
        for (i, t) in enumerate(t_list)
            println("Time in χ(t) is $(t)")
            println("Current time is $(now())")
            U_H1t = exp(- 1im * t * ham)
            U_H2t = adjoint(U_H1t) #exp(itD) *
            pauli_OA_t = U_H1t*pauli_OA*U_H2t

            op = pauli_OA_t

            for l in 0:4^(nsys_spin)-1
                pauli_number_basis = basis_label_spin(l, nsys_spin)
                pauli_number = 0
                for pauli_type in pauli_number_basis
                    if pauli_type != 0
                        pauli_number += 1
                    end
                end
                tempc = tr(spin_hilbert_basis_arr[l+1] * op) / ndim
                c_basis_arr[l+1,i,Nreal] = tempc

                Pn_arr[pauli_number+1,i,Nreal] += abs2(tempc)
                Qn_arr[pauli_number+1,i,Nreal] += tempc^2
            end
            if i <= 3
                display(c_basis_arr[:,i,Nreal])
                display(Pn_arr[:,i,Nreal])
                
            end

            @test sum(Pn_arr[:,i,1]) ≈ 1.0 + 0.0im
            @test dot(c_basis_arr[:,i,1], c_basis_arr[:,i,1]) ≈ 1.0 + 0.0im

        end
        display(Pn_arr[:,:,1])

        JLD2.save(joinpath(data_dir, "Size_N$(nsys_spin)_random$(Nrealization)_W$(W_disorder).jld")
            ,"Pn_arr", Pn_arr, "Qn_arr", Qn_arr, "t_list", t_list)
    end
    
    return c_basis_arr, Pn_arr, Qn_arr
end

function A_polynomial(xi::Array{Float64,1})
    return (-xi[1]^2 + xi[2]^2 - 4*xi[2]*xi[3] + xi[3]^2)
end
    
#####################################################
# If nsys_spin = 11
# spin_hilbert_basis_arr memory will be 256 Gbyte
# Therefore the maximum choice is nsys_spin = 9
#####################################################

data_dir = "OA_ED_data"
nsys_spin = 6
k = 2
Nrealization=1
Delta = 1.0
W_disorder = 10.0
OA_type = 3
q = 2
# load_flag = true
load_flag = false
#############################################################################
# Can be comment out
spin_hilbert_basis_arr, spin_single_site_basis_arr = prepare_spin_basis(nsys_spin, data_dir, load_flag)
#############################################################################


# Calculate the shadow norm using the size distribution function P(n)
##################################
t_list = collect(0.0:0.1:20.0) .+ 0.0im
nonzeroindex = collect(1:nsys_spin+1)
label_n = Array{String, 2}(undef, 1, length(nonzeroindex))
title_plot = L"$N_{\textrm{p}}=%$(nsys_spin),\textrm{Rand}=%$(Nrealization),W=%$(W_disorder)$"
filename_extension = "N$(nsys_spin)_random$(Nrealization)_W$(W_disorder)_OAtype$(OA_type)"
for ll in 1:length(nonzeroindex)
    label_n[1,ll] = "\$ n=$(ll-1) \$"
end

c_basis_arr, Pn_arr, Qn_arr =  sizeDis_windsizeDis(nsys_spin, W_disorder, Delta, t_list,Nrealization,spin_hilbert_basis_arr, spin_single_site_basis_arr, OA_type)
Pn_arr_aver = (sum(Pn_arr[:,:,:], dims=3) /Nrealization)[:,:,1]
Qn_arr_aver = (sum(Qn_arr[:,:,:], dims=3) /Nrealization)[:,:,1]
c_basis_arr_aver = (sum(c_basis_arr[:,:,:], dims=3) /Nrealization)[:,:,1]

OA_sh_t_arr = zeros(nsys_spin+1, length(t_list))
for i in 0:nsys_spin
    OA_sh_t_arr[i+1, :] = Pn_arr_aver[i+1, :] * (q+1)^(-1.0*i)
end
OA_sh_t = sum(OA_sh_t_arr, dims=1)[1,:]
OA_sh_t = 1.0 ./ OA_sh_t

display(OA_sh_t)

@test imag(sum(Pn_arr)) == 0.0

plot(real.(t_list), log.(q+1, OA_sh_t), xlabel=L"$t$", ylabel=L"$\log_3 ||O||_\textrm{sh}^2$", title=L"ED, $N=%$(nsys_spin),k=%$(k),W=%$(W_disorder),\textrm{Nrand}=%$(Nrealization),O_A=\prod\sigma_%$(OA_type)$",titlefontsize=13)
savefig("figs_ED/OA_sh_t_ED_$(filename_extension).pdf")