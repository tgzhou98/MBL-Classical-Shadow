using LaTeXStrings
using Distributions
using JLD2
using Dates
using Statistics
using Plots
using LsqFit

include("itensor_setup.jl")

function reconstruct(nsite::Int64, cutoff::Float64, tau::Float64, ttotal::Float64,
    Nrealization::Int64, Delta::Float64, W_disorder::Float64, k_len::Int64, total_dirname::String, state_name::String)

    ################################################
    # The position of operator O_A with length k, which we want to apply classical shadow protocal

    # Choice 1
    #=
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len÷2
        op_k_pos_arr[2*(i-1)+1] = nsite ÷ 2 + 1 - i
        op_k_pos_arr[2*(i-1)+2] = nsite ÷ 2 + i
    end
    =#

    # Choice 2
    op_k_pos_arr = ones(Int64, k_len)
    for i in 1:k_len
        op_k_pos_arr[i] = nsite ÷ 2 + i
    end
    ################################################

    clif_num = 24

    file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"
    # Site basis
    s = siteinds("S=1/2", nsite)

    # Load single site random clifford gates and random field h_arr used in the sampling stage
    clif_ind_layer_rand = JLD2.load("$(total_dirname)/sample_result_$(file_extension)_config.jld2", "clif_ind_layer_rand")
    h_arr = JLD2.load("$(total_dirname)/sample_result_$(file_extension)_config.jld2", "h_arr")
    println("h_arr")
    display(h_arr)

    # Load samples
    shadow_sample_arr = JLD2.load("$(total_dirname)/sample_result_$(file_extension).jld2", "shadow_sample_arr")
    display(shadow_sample_arr)

    OA_expect_k_realizationall = zeros(ComplexF64, Nrealization, k_len)

    # Construction of inverse MBL gate
    gates_MBL_adj = ITensor[]
    for j in 1:(nsite - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj = (Delta * op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)) +
            h_arr[j] / 2 * op("Sz", s1) * op("Id", s2) + h_arr[j + 1] / 2 * op("Id", s1) * op("Sz", s2)
        if j == 1
            hj += h_arr[j] / 2 * (op("Sz", s1) * op("Id", s2))
        elseif j == nsite - 1
            hj += h_arr[j + 1] / 2 * (op("Sz", s2) * op("Id", s1))
        end
        Gj = exp(im * tau / 2 * hj)
        push!(gates_MBL_adj, Gj)
    end
    # Second order trotter decomposition
    append!(gates_MBL_adj, reverse(gates_MBL_adj))

    
    for nreal in 1:Nrealization
        # Samples are represented by the product state
        sample_arr = shadow_sample_arr[nreal,:]
        states = [isodd(sample_arr[n]) ? "Up" : "Dn" for n in 1:nsite]
        psi = MPS(s, states)

        clif_ind_layer_init = clif_ind_layer_rand[nreal,:,1]
        clif_ind_layer_end  = clif_ind_layer_rand[nreal,:,2]


        # Init on-site clifford layer for reconstruction, which is equal to the ending layer for sampling
        gates_clif_rev_init = ITensor[]
        for j in 1:nsite
            clif_ind = clif_ind_layer_end[j]
            push!(gates_clif_rev_init, (op("dagcli$(clif_ind)", s[j])))
        end
        # display(gates_clif_init)

        # Ending on-site clifford layer for reconstruction, which is equal to the initial layer for sampling
        gates_clif_rev_end = ITensor[]
        for j in 1:nsite
            clif_ind = clif_ind_layer_init[j]
            push!(gates_clif_rev_end, (op("dagcli$(clif_ind)", s[j])))
        end

        ####################################################
        
        # Initial on-site clifford layer
        psi = apply(gates_clif_rev_init, psi; cutoff)
        # Reverse MBL quantum circuit
        for (i, t) in enumerate(0.0:tau:ttotal)
            t≈ttotal && break

            psi = apply(gates_MBL_adj, psi; cutoff)
            normalize!(psi)
        end
        # Ending on-site clifford layer
        psi = apply(gates_clif_rev_end, psi; cutoff)
        normalize!(psi)
        ####################################################

        # Calculation of the expectation value of operator O_A
        cutoff_sz_observation = 1E-22
        OA_expect_k = zeros(ComplexF64, k_len)
        psi_final = deepcopy(psi)
        for (k_ind, k_pos) in enumerate(op_k_pos_arr)

            psi_final = apply(op("Sz", s[k_pos]), psi_final; cutoff_sz_observation)
            OA_expect_k[k_ind] = inner(psi, psi_final) * 2^(k_ind)
        end
        OA_expect_k_realizationall[nreal, :] = OA_expect_k


        if nreal % 500 == 1
            println("nreal=$(nreal), now=$(now())")
            println("$(file_extension)")
            println("OA_k =")
            display((OA_expect_k))

            JLD2.save("$(total_dirname)/$(file_extension)_reconstruction_result.jld2", "OA_expect_k_realizationall", OA_expect_k_realizationall)
        end
        if nreal == 2
            println("nreal 2, Time = $(now())")
            println("OA_k =")
            display((OA_expect_k))
        end

    end
    JLD2.save("$(total_dirname)/$(file_extension)_reconstruction_result.jld2", "OA_expect_k_realizationall", OA_expect_k_realizationall)
    
    return OA_expect_k_realizationall
end


function main_reconstruction()
    nsite = 30                       # Site Number
    cutoff = 1E-14                   # cutoff δ_χ for MPS
    tau = 0.1                        # Trotter decomposition time step
    ttotal = 2.0                     # Total evolution time of MBL Hamiltonian
    Delta = 1.0                      # XXZ parameter, XX+YY+Delta*ZZ
    Nrealization = 50000             # Total random realization number            
    W_disorder_arr = [5.0, 10.0]     # The disorder potential W*Z for the XXZ model                    
    state_name_arr = ["GHZ", "ZXZ"]  # The unknown initial state for classical shadow protocal                    
    k_len = 8                        # The maximal length of operator O_A
    maxdim_val = 200                 # The maximal bond dimension for the shadow norm calculation


    for state_name in state_name_arr
        for W_disorder in W_disorder_arr
            total_dirname = "tJ$(ttotal)_sample_$(state_name)"
            if !isdir(total_dirname)
                mkdir(total_dirname)
            end

            # Load shadow norm from previous data
            file_extension_OA_sh = "N$(nsite)_W$(W_disorder)_tau$(tau)"
            file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"
            OA_sh_t_all = JLD2.load("$(total_dirname)/OA_sh_fixOpdata_allk_type_$(file_extension_OA_sh)_chi$(maxdim_val).jld2", "OA_sh_t_all")

            # reconstruct the ⟨O_A⟩ using the classical shadow
            OA_expect_k_realizationall = reconstruct(nsite, cutoff, tau, ttotal,
                Nrealization, Delta, W_disorder, k_len, total_dirname, state_name)

            t_len = size(OA_sh_t_all)[1]
            OA_expect_all_anystate = OA_expect_k_realizationall
            # Need to divide the ||O_A||_sh^2 for the final result
            for i in 1:k_len
                OA_expect_all_anystate[:,i] = OA_expect_all_anystate[:,i] / OA_sh_t_all[t_len,i]
            end

            OA_expect_anystate_k_result_mean = real.(mean(OA_expect_all_anystate, dims=1)[1,:])
            OA_expect_anystate_k_result_var =  real.(var(OA_expect_all_anystate, dims=1)[1,:])

            # Random realization number
            M = Nrealization
            # 2 Times standard deviation
            std2OA = 2*sqrt.(OA_expect_anystate_k_result_var/M)

            OA_expect = ((OA_expect_anystate_k_result_mean))
            println("<O_A> = $(OA_expect_anystate_k_result_mean)")
            println("var O_A = $(OA_expect_anystate_k_result_var)")
            println("std O_A = $(2*sqrt.(OA_expect_anystate_k_result_var/M))")
            println("(|OA|_sh^2/M)  =$(  (1.0 / (M)) ./OA_sh_t_all[t_len,:] )")
            println("(|OA|_sh^2) = $(1.0 ./OA_sh_t_all[t_len,:] )")
            println("(log(|OA|_sh^2)) = $( -log.(3, OA_sh_t_all[t_len,:]) )")

            file_extension = "N$(nsite)_W$(W_disorder)_Nrand$(Nrealization)_tau$(tau)"

            scatter(OA_expect, yerr=std2OA, title="$(state_name), Rand=$(Nrealization)", ylabel=L"$\langle Z^{\otimes k} \rangle$", xlabel=L"k")
            savefig("$(total_dirname)/$(file_extension)_$(state_name)_expect.pdf")


            JLD2.save("$(total_dirname)/$(file_extension)_reconstruction_plotdata.jld2", "OA_expect", OA_expect, "std2OA", std2OA)
            ########################################################

            ############################################################
            # Plot the shadow norm on the log_3 basis
            OA_sh_log3_t_all = -log.(3, OA_sh_t_all)
            t_arr = collect(0.0:tau:ttotal)
            t_index_last = length(t_arr)

            logOA_k_slope_arr = zeros(t_index_last)
            @. model(x, p) = p[1] + (x*p[2])

            for i in 1:t_index_last
                xdata = collect(1:k_len)
                ydata = OA_sh_log3_t_all[i,1:k_len]
                p0 = [0.5, 0.5]
                fit = curve_fit(model, xdata, ydata, p0)
                b, slope = coef(fit)
                logOA_k_slope_arr[i] = slope
            end
            fontsize = 12
            plot(t_arr[1:t_index_last], logOA_k_slope_arr, xlabel=L"$t$", ylabel=L"$\mathrm{d} \log_3 ||O||_\textrm{sh}^2 / \mathrm{d}k$", title=L"TEBD,$N=%$(nsite),W=%$(W_disorder),\tau=%$(tau),\chi=%$(maxdim_val)$", titlefontsize=fontsize)
            savefig("$(total_dirname)/fitslope_OA_sh_MBL_$(file_extension)_allk_averOA.pdf")
            savefig("$(total_dirname)/fitslope_OA_sh_MBL_$(file_extension)_allk_averOA.png")
        
        end
    end
end

main_reconstruction()