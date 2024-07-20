using Plots
using LaTeXStrings
using Distributions
using JLD2
using Dates

include("itensor_setup.jl")

function OAnorm_for_sample(t_arr::Array{Float64, 1}, k_arr::Array{Int64, 1}, Delta::Float64, h_arr::Array{Float64, 1}, nsite::Int64, W_disorder::Float64, Nrealization::Int64,
    maxdim_val::Int64, cutoff_val::Float64, state_name::String, file_extension::String)

    ttotal = last(t_arr)
    total_dirname = "tJ$(ttotal)_sample_$(state_name)"

    s = siteinds("doubleOpState", nsite)

    # Clarify data arr
    OA_sh_t_all = zeros(length(t_arr), length(k_arr))
    if length(t_arr) == 1
        tau = 0.0
    else
        tau = t_arr[2] - t_arr[1]
    end
    tlen = length(t_arr)
    ttotal = t_arr[tlen]
    
    # Prepare all target state
    psi_target_arr = MPS[]

    k_len = length(k_arr)
    ################################################
    # Old position
    #=
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len÷2
        op_k_pos_arr[2*(i-1)+1] = nsite ÷ 2 + 1 - i
        op_k_pos_arr[2*(i-1)+2] = nsite ÷ 2 + i
    end
    =#
    ################################################

    ################################################
    # New position
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len
        op_k_pos_arr[i] = nsite ÷ 2 + i
    end
    ################################################

    targetstate = ["IIstate" for n=1:nsite]

    psi_target_klen_map = Int64[]
    k_len = length(k_arr)
    OA_weight_each_klen = [1.0 for k in 1:k_len]
    # Construction of the initial state for calculate ||O_A||_sh^2
    # If the site k is support by nontrivial operator, then the site k in targetstate is initialized in "Occstate"
    for k in 1:k_len
        targetstate[(op_k_pos_arr[k])] = "Occstate"
        psi_target = productMPS(s,targetstate)
        push!(psi_target_arr, psi_target)
        push!(psi_target_klen_map, k)
    end

    # Construction of the Double operator state Hamiltonian
    gates = ITensor[]
    for j in 1:(nsite - 1)
        s1 = s[j]
        s2 = s[j + 1]
        # All op is Pauli matrix, 1/4*(XX+YY+ZZ) - h *Z/4 - h *Z/4
        hamj =  1 / 4 * (Delta * op("SzL1", s1) * op("SzL1", s2) +
            1 / 2 * op("S+L1", s1) * op("S-L1", s2) +
            1 / 2 * op("S-L1", s1) * op("S+L1", s2) + 
            h_arr[j] * op("SzL1", s1) * op("IdL1", s2) + h_arr[j + 1] * op("IdL1", s1) * op("SzL1", s2))
        hamj -= 1 / 4 * (Delta * op("SzR1", s1) * op("SzR1", s2) +
            1 / 2 * op("S+R1", s1) * op("S-R1", s2) +
            1 / 2 * op("S-R1", s1) * op("S+R1", s2) - 
            h_arr[j] * op("SzR1", s1) * op("IdR1", s2) - h_arr[j + 1] * op("IdR1", s1) * op("SzR1", s2))
        hamj += 1 / 4 * (Delta * op("SzL2", s1) * op("SzL2", s2) +
            1 / 2 * op("S+L2", s1) * op("S-L2", s2) +
            1 / 2 * op("S-L2", s1) * op("S+L2", s2) + 
            h_arr[j] * op("SzL2", s1) * op("IdL2", s2) + h_arr[j + 1] * op("IdL2", s1) * op("SzL2", s2))
        hamj -= 1 / 4 * (Delta * op("SzR2", s2) * op("SzR2", s1) +
            1 / 2 * op("S+R2", s2) * op("S-R2", s1) +
            1 / 2 * op("S-R2", s2) * op("S+R2", s1) - 
            h_arr[j] * op("SzR2", s1) * op("IdR2", s2) - h_arr[j + 1] * op("IdR2", s1) * op("SzR2", s2))

        # Deal with the boundary term
        if j == 1
            hamj += h_arr[j] / 4 * (op("SzL1", s1) * op("IdL1", s2) + op("SzR1", s1) * op("IdR1", s2)
                        + op("SzL2", s1) * op("IdL2", s2) + op("SzR2", s1) * op("IdR2", s2))
        elseif j == nsite - 1
            hamj += h_arr[j + 1] / 4 * (op("SzL1", s2) * op("IdL1", s1) + op("SzR1", s2) * op("IdR1", s1)
                        + op("SzL2", s2) * op("IdL2", s1) + op("SzR2", s2) * op("IdR2", s1))
        end
        Gj = exp(im * tau / 2 * hamj)
        push!(gates, Gj)
    end
    # Include gates in reverse order too, second order trotter
    append!(gates, reverse(gates))

    initstate = ["initstate" for n=1:nsite]
    psi = productMPS(s,initstate)


    for (i, t) in enumerate(t_arr)
        println("t=$(t)")

        for (ll, k_len_each) in enumerate(psi_target_klen_map)
            # The shadow norm is calculate by the inner product between the evoloved state psi and the initial state psi_target
            psi_target = psi_target_arr[ll]

            OA_sh_t = inner(psi_target,psi)
            OA_sh_t = real(OA_sh_t)
            OA_sh_t_all[i,k_len_each] += OA_sh_t * OA_weight_each_klen[k_len_each]
            if ll == 1
                println("k=$(k_len_each),OA_sh_t=$(OA_sh_t)")
            end
        end

        t≈ttotal && break

        println("OA_sh_t_all in t step $(i)")
        display(OA_sh_t_all[i,:])

        psi = apply(gates, psi; cutoff=cutoff_val, maxdim=maxdim_val)
        JLD2.save("$(total_dirname)/OA_sh_fixOpdata_allk_type_$(file_extension)_chi$(maxdim_val).jld2", "t_arr", t_arr, 
            "OA_sh_t_all", OA_sh_t_all, "h_arr", h_arr)

        println("Current time is $(now())")
    end


    # Up to here OA_sh is still λ_A
    # After average we can perform inverse
    OA_sh_log3_t_all = -log.(3, OA_sh_t_all)


    JLD2.save("$(total_dirname)/OA_sh_fixOpdata_allk_type_$(file_extension)_chi$(maxdim_val).jld2", "t_arr", t_arr, 
        "OA_sh_t_all", OA_sh_t_all, "h_arr", h_arr, "OA_sh_log3_t_all", OA_sh_log3_t_all)
    
    label_k = Array{String, 2}(undef, 1, length(k_arr))
    for ll in 1:length(k_arr)
        label_k[1,ll] = "\$ k=$(k_arr[ll]) \$"
    end

    fontsize = 12
    plot(t_arr, OA_sh_log3_t_all[:,:], label=label_k, xlabel=L"$t$", ylabel=L"$\log_3 ||O||_\textrm{sh}^2$", title=L"TEBD,$N=%$(nsite),W=%$(W_disorder),\textrm{Nrand}=%$(Nrealization),\tau=%$(tau),\chi=%$(maxdim_val)$", titlefontsize=fontsize)
    savefig("$(total_dirname)/OA_sh_fixOpMBL_$(file_extension)_allk_averOA_chi$(maxdim_val).pdf")
    savefig("$(total_dirname)/OA_sh_fixOpMBL_$(file_extension)_allk_averOA_chi$(maxdim_val).png")
    
    return OA_sh_t_all, h_arr
    
end


function run_OA_sh_fixedOA()
    nsite = 30                       # Site Number
    cutoff_val = 1E-28               # cutoff for shadow norm MPS
    maxdim_val = 200                 # The maximal bond dimension for the shadow norm calculation
    tau = 0.1                        # Trotter decomposition time step
    ttotal = 2.0                     # Total evolution time of MBL Hamiltonian
    Delta = 1.0                      # XXZ parameter, XX+YY+Delta*ZZ
    Nrealization = 50000             # Total random realization number used in the sample and reconstruction stages           
    W_disorder_arr = [5.0, 10.0]     # The disorder potential W*Z for the XXZ model                    
    state_name_arr = ["GHZ", "ZXZ"]  # The unknown initial state for classical shadow protocal                    
    k_len = 8                        # The maximal length of operator O_A

    t_arr = collect(0.0:tau:ttotal)  # time array
    k_arr = collect(1:1:k_len)       # OA length k array

    for state_name in state_name_arr

        for W_disorder in W_disorder_arr
            file_extension = "N$(nsite)_W$(W_disorder)_tau$(tau)"
            println("$(file_extension)")

            total_dirname = "tJ$(ttotal)_sample_$(state_name)"
            if !isdir(total_dirname)
                mkdir(total_dirname)
            end

            file_extension_harr = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"
            h_arr = JLD2.load("$(total_dirname)/sample_result_$(file_extension_harr)_config.jld2", "h_arr")

            display(h_arr)

            OA_sh_t_all, h_arr = OAnorm_for_sample(t_arr, k_arr, Delta, h_arr, nsite, W_disorder, Nrealization, maxdim_val, cutoff_val, state_name, file_extension)
        end
    end

    
end

 run_OA_sh_fixedOA()